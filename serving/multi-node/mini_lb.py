import time
import uuid
import aiohttp
import asyncio
import logging
import argparse
import uvicorn
from threading import Lock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WorkerNode:
    """Represents a worker node with its status and request count."""

    url: str
    is_healthy: bool = False
    current_requests: int = 0
    last_health_check: float = field(default_factory=time.time)

    def can_accept_request(self, max_requests: int) -> bool:
        """Check if this worker can accept a new request."""
        return self.is_healthy and self.current_requests < max_requests


@dataclass
class QueuedRequest:
    """Represents a queued request waiting for an available worker."""

    request_id: str
    endpoint: str
    request_data: Dict[str, Any]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


class LoadBalancer:
    """Load balancer that manages worker nodes and distributes requests."""

    def _safe_create_task(self, coro, task_name="task"):
        """Safely create a task only if event loop is running and not closed."""
        try:
            loop = asyncio.get_running_loop()
            if not loop.is_closed():
                return asyncio.create_task(coro)
            else:
                logger.debug(f"Cannot create {task_name}: event loop is closed")
                return None
        except RuntimeError:
            logger.debug(f"Cannot create {task_name}: no running event loop")
            return None

    def __init__(
        self,
        worker_urls: List[str],
        max_requests_per_worker: int,
        max_queue_size: int = 100,
    ):
        self.workers = [WorkerNode(url) for url in worker_urls]
        self.max_requests_per_worker = max_requests_per_worker
        self.current_worker_index = 0
        self.lock = Lock()

        # Configure connector for high-volume requests
        connector = aiohttp.TCPConnector(
            limit=16384,  # Total connection pool size
            limit_per_host=16384,  # Connections per host
            keepalive_timeout=120,  # Keep connections alive for 2 minutes
            enable_cleanup_closed=True,  # Clean up closed connections
            use_dns_cache=True,  # Cache DNS lookups
            ttl_dns_cache=300,  # DNS cache TTL
            force_close=False,  # Reuse connections
        )

        # Create session with optimized settings for high volume
        self.client = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=300,  # Increased total timeout for generation
                connect=10,  # Connection timeout
                sock_read=120,  # Socket read timeout
            ),
            connector=connector,
            # Skip auto-decompression to save CPU
            auto_decompress=True,
            # Increase read buffer size
            read_bufsize=2**16,  # 64KB
        )

        # Request queue
        self.request_queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self.queue_processor_task: Optional[asyncio.Task] = None
        self._processing_queue = False

    async def health_check_worker(self, worker: WorkerNode) -> bool:
        """Check if a worker is healthy by calling its /health endpoint."""
        try:
            async with self.client.get(f"{worker.url}/health", timeout=5.0) as response:
                is_healthy = response.status == 200
                worker.is_healthy = is_healthy
                worker.last_health_check = time.time()
                logger.debug(
                    f"Health check for {worker.url}: {'healthy' if is_healthy else 'unhealthy'}"
                )
                return is_healthy
        except Exception as e:
            logger.debug(f"Health check failed for {worker.url}: {e}")
            worker.is_healthy = False
            worker.last_health_check = time.time()
            return False

    async def check_all_workers_health(self):
        """Check health of all workers."""
        tasks = [self.health_check_worker(worker) for worker in self.workers]
        await asyncio.gather(*tasks, return_exceptions=True)

        healthy_count = sum(1 for worker in self.workers if worker.is_healthy)
        logger.debug(
            f"Health check complete: {healthy_count}/{len(self.workers)} workers healthy"
        )

        # Process queue in case workers became healthy and can handle queued requests
        if healthy_count > 0:
            self._safe_create_task(
                self._process_next_queued_request(),
                "queue processor after health check",
            )

    def get_next_available_worker(self) -> Optional[WorkerNode]:
        """Get the next available worker using round-robin with request limits."""
        with self.lock:
            attempts = 0
            while attempts < len(self.workers):
                worker = self.workers[self.current_worker_index]

                if worker.can_accept_request(self.max_requests_per_worker):
                    worker.current_requests += 1
                    selected_worker = worker
                    # Move to next worker for round-robin
                    self.current_worker_index = (self.current_worker_index + 1) % len(
                        self.workers
                    )
                    logger.debug(
                        f"Selected worker {selected_worker.url} (requests: {selected_worker.current_requests}/{self.max_requests_per_worker})"
                    )
                    return selected_worker

                # Try next worker
                self.current_worker_index = (self.current_worker_index + 1) % len(
                    self.workers
                )
                attempts += 1

            logger.debug("No available workers found")
            return None

    def release_worker(self, worker: WorkerNode):
        """Release a worker after request completion."""
        with self.lock:
            if worker.current_requests > 0:
                worker.current_requests -= 1
                logger.debug(
                    f"Released worker {worker.url} (requests: {worker.current_requests}/{self.max_requests_per_worker})"
                )

        # Trigger queue processing when a worker becomes available
        self._safe_create_task(
            self._process_next_queued_request(), "queue processor after worker release"
        )

    async def start_queue_processor(self):
        """Initialize queue processing - now event-driven, no background task needed."""
        logger.info("Queue processor initialized (event-driven)")
        # Process any existing queued requests
        await self._process_next_queued_request()

    async def stop_queue_processor(self):
        """Stop the queue processor task."""
        if self.queue_processor_task and not self.queue_processor_task.done():
            self.queue_processor_task.cancel()
            try:
                await self.queue_processor_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped request queue processor")

    async def _process_next_queued_request(self):
        """Process the next queued request if a worker is available."""
        # Prevent multiple concurrent queue processing
        if self._processing_queue:
            return

        self._processing_queue = True
        try:
            # Process as many queued requests as we have available workers
            while not self.request_queue.empty():
                worker = self.get_next_available_worker()
                if not worker:
                    break  # No available workers, stop processing

                try:
                    # Get next queued request (non-blocking)
                    queued_request = self.request_queue.get_nowait()

                    # Process the request immediately (if event loop is still running)
                    task = self._safe_create_task(
                        self._handle_queued_request(worker, queued_request),
                        "queued request handler",
                    )
                    if task is None:
                        # Event loop is closed or no running loop, release the worker and break
                        self.release_worker(worker)
                        break

                except asyncio.QueueEmpty:
                    break  # No more queued requests

        finally:
            self._processing_queue = False

    async def _handle_queued_request(
        self, worker: WorkerNode, queued_request: QueuedRequest
    ):
        """Handle a single queued request."""
        try:
            result = await self.forward_request(
                worker, queued_request.endpoint, queued_request.request_data
            )
            queued_request.future.set_result(result)
            logger.debug(
                f"Processed queued request {queued_request.request_id} after {time.time() - queued_request.timestamp:.2f}s"
            )
        except Exception as e:
            queued_request.future.set_exception(e)
            logger.error(
                f"Failed to process queued request {queued_request.request_id}: {e}"
            )
        finally:
            self.request_queue.task_done()

    async def _queue_request(
        self, endpoint: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Queue a request when no workers are available."""
        request_id = str(uuid.uuid4())
        future = asyncio.Future()

        queued_request = QueuedRequest(
            request_id=request_id,
            endpoint=endpoint,
            request_data=request_data,
            future=future,
        )

        try:
            # Try to add to queue (non-blocking)
            self.request_queue.put_nowait(queued_request)
            logger.debug(
                f"Queued request {request_id} (queue size: {self.request_queue.qsize()})"
            )

            # Immediately try to process queued requests in case workers are available
            self._safe_create_task(
                self._process_next_queued_request(),
                "queue processor after request queued",
            )

            # Wait for the result
            return await future

        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Request queue is full")

    async def forward_request(
        self, worker: WorkerNode, endpoint: str, request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forward a request to a specific worker endpoint with retry logic."""
        max_retries = 5  # Increased retries for high-volume scenarios
        base_delay = 0.1  # Shorter initial delay
        max_delay = 5.0  # Cap maximum delay

        try:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    # Use session timeout (already configured in __init__)
                    async with self.client.post(
                        f"{worker.url}{endpoint}",
                        json=request_data,
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.warning(
                                f"HTTP error from worker {worker.url}{endpoint}: {response.status} (attempt {attempt + 1}/{max_retries})"
                            )

                            # Don't retry for client errors (4xx), only server errors (5xx)
                            if response.status < 500:
                                raise HTTPException(
                                    status_code=response.status,
                                    detail=f"Worker error: {error_text}",
                                )

                            # For server errors, retry if we have attempts left
                            if attempt < max_retries - 1:
                                delay = min(base_delay * (2**attempt), max_delay)
                                logger.info(
                                    f"Retrying HTTP {response.status} in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(delay)
                                continue
                            else:
                                raise HTTPException(
                                    status_code=response.status,
                                    detail=f"Worker error after {max_retries} attempts: {error_text}",
                                )

                        return await response.json()

                except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                    last_exception = e
                    logger.warning(
                        f"Timeout from worker {worker.url}{endpoint} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}"
                    )

                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.info(
                            f"Retrying timeout in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(
                            status_code=504,
                            detail=f"Worker timeout after {max_retries} attempts",
                        )

                except aiohttp.ClientConnectionError as e:
                    last_exception = e
                    logger.warning(
                        f"Connection error to worker {worker.url}{endpoint} (attempt {attempt + 1}/{max_retries}): {e}"
                    )

                    # Mark worker as unhealthy on connection errors
                    worker.is_healthy = False

                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.info(
                            f"Retrying connection error in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Connection error after {max_retries} attempts",
                        )

                except aiohttp.ClientOSError as e:
                    last_exception = e
                    logger.warning(
                        f"OS error to worker {worker.url}{endpoint} (attempt {attempt + 1}/{max_retries}): {e}"
                    )

                    # OS errors (like too many open files) should be retried with longer delays
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (3**attempt), max_delay * 2
                        )  # Longer backoff for OS errors
                        logger.info(
                            f"Retrying OS error in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(
                            status_code=502,
                            detail=f"OS error after {max_retries} attempts: {str(e)}",
                        )

                except aiohttp.ClientError as e:
                    last_exception = e
                    logger.warning(
                        f"Client error to worker {worker.url}{endpoint} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                    )

                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.info(
                            f"Retrying client error in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Client error after {max_retries} attempts",
                        )

                except RuntimeError as e:
                    if "Event loop is closed" in str(e):
                        # Event loop is closing, don't retry - fail fast
                        logger.error(
                            f"Event loop closed during request to {worker.url}{endpoint}, failing immediately"
                        )
                        raise HTTPException(
                            status_code=503, detail="Service shutting down"
                        )
                    else:
                        # Other RuntimeErrors should be retried
                        last_exception = e
                        logger.error(
                            f"Runtime error to worker {worker.url}{endpoint} (attempt {attempt + 1}/{max_retries}): {e}"
                        )

                        if attempt < max_retries - 1:
                            delay = min(base_delay * (2**attempt), max_delay)
                            logger.info(
                                f"Retrying runtime error in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Runtime error after {max_retries} attempts",
                            )

                except Exception as e:
                    last_exception = e
                    logger.error(
                        f"Unexpected error to worker {worker.url}{endpoint} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
                    )

                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2**attempt), max_delay)
                        logger.info(
                            f"Retrying unexpected error in {delay:.2f}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        try:
                            await asyncio.sleep(delay)
                        except RuntimeError as sleep_error:
                            if "Event loop is closed" in str(sleep_error):
                                logger.error(
                                    "Event loop closed during retry delay, failing immediately"
                                )
                                raise HTTPException(
                                    status_code=503, detail="Service shutting down"
                                )
                            raise
                        continue
                    else:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Unexpected error after {max_retries} attempts",
                        )

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        finally:
            self.release_worker(worker)

    async def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standalone function to handle generate requests."""
        worker = self.get_next_available_worker()
        if worker:
            return await self.forward_request(worker, "/generate", request_data)
        else:
            # Queue the request if no workers are available
            return await self._queue_request("/generate", request_data)

    async def chat_completions(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standalone function to handle chat completions requests."""
        worker = self.get_next_available_worker()
        if worker:
            return await self.forward_request(
                worker, "/v1/chat/completions", request_data
            )
        else:
            # Queue the request if no workers are available
            return await self._queue_request("/v1/chat/completions", request_data)

    async def completions(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standalone function to handle completions requests."""
        worker = self.get_next_available_worker()
        if worker:
            return await self.forward_request(worker, "/v1/completions", request_data)
        else:
            # Queue the request if no workers are available
            return await self._queue_request("/v1/completions", request_data)

    async def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        status = {
            "workers": [],
            "total_workers": len(self.workers),
            "healthy_workers": sum(1 for w in self.workers if w.is_healthy),
            "max_requests_per_worker": self.max_requests_per_worker,
            "queue_size": self.request_queue.qsize(),
            "queue_max_size": self.request_queue.maxsize,
        }

        for worker in self.workers:
            status["workers"].append(
                {
                    "url": worker.url,
                    "healthy": worker.is_healthy,
                    "current_requests": worker.current_requests,
                    "last_health_check": worker.last_health_check,
                }
            )

        return status

    async def close(self):
        """Clean up resources."""
        logger.info("Shutting down load balancer...")

        # Stop queue processor
        await self.stop_queue_processor()

        # Close client session with proper cleanup
        if self.client and not self.client.closed:
            # Wait for pending requests to complete (with timeout)
            try:
                await asyncio.wait_for(self.client.close(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Client session close timed out, forcing closure")
                # Force close if it takes too long
                await self.client.close()

        logger.info("Load balancer shutdown complete")

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for monitoring."""
        if not self.client or self.client.closed:
            return {"error": "Client session not available"}

        connector = self.client.connector
        if hasattr(connector, "_conns"):
            # Get connection pool statistics
            stats = {
                "total_connections": len(connector._conns),
                "connection_limit": connector.limit,
                "connection_limit_per_host": connector.limit_per_host,
                "acquired_connections": getattr(connector, "_acquired", 0),
                "closed": self.client.closed,
            }

            # Get per-host connection counts
            host_connections = {}
            for key, connections in connector._conns.items():
                host_connections[str(key)] = len(connections)
            stats["connections_per_host"] = host_connections

            return stats

        return {"error": "Connection statistics not available"}

    async def cleanup_connections(self):
        """Manually trigger connection cleanup."""
        if self.client and not self.client.closed:
            connector = self.client.connector
            if hasattr(connector, "_cleanup_closed_transports"):
                # Trigger cleanup of closed connections
                await connector._cleanup_closed_transports()
                logger.debug("Cleaned up closed connections")


# Global load balancer instance
load_balancer: Optional[LoadBalancer] = None
health_check_task: Optional[asyncio.Task] = None


async def periodic_health_check():
    """Periodically check worker health and clean up connections."""
    cleanup_counter = 0
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            if load_balancer:
                await load_balancer.check_all_workers_health()

                # Clean up connections every 5 health checks (2.5 minutes)
                cleanup_counter += 1
                if cleanup_counter >= 5:
                    await load_balancer.cleanup_connections()
                    cleanup_counter = 0

                    # Log connection stats periodically
                    stats = await load_balancer.get_connection_stats()
                    if "error" not in stats:
                        logger.info(
                            f"Connection stats: {stats['total_connections']} total, {stats.get('acquired_connections', 0)} acquired"
                        )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic health check: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global health_check_task

    # Startup
    if load_balancer:
        # Initial health check
        await load_balancer.check_all_workers_health()

        # Start periodic health checking
        health_check_task = asyncio.create_task(periodic_health_check())

        # Initialize queue processor (event-driven, no background task)
        await load_balancer.start_queue_processor()

    yield

    # Shutdown
    if health_check_task:
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass

    if load_balancer:
        await load_balancer.close()


# FastAPI app
app = FastAPI(title="Simple Load Balancer", version="1.0.0", lifespan=lifespan)


@app.post("/generate")
async def generate_endpoint(request: Request):
    """Forward generate requests to available workers."""
    if not load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")

    # Get request data
    try:
        request_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Forward request using standalone function
    return await load_balancer.generate(request_data)


@app.post("/v1/chat/completions")
async def chat_completions_endpoint(request: Request):
    """Forward chat completions requests to available workers."""
    if not load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")

    # Get request data
    try:
        request_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Forward request using standalone function
    return await load_balancer.chat_completions(request_data)


@app.post("/v1/completions")
async def completions_endpoint(request: Request):
    """Forward completions requests to available workers."""
    if not load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")

    # Get request data
    try:
        request_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Forward request using standalone function
    return await load_balancer.completions(request_data)


@app.get("/health")
async def health():
    """Health endpoint for the load balancer itself."""
    if not load_balancer:
        return {"status": "error", "message": "Load balancer not initialized"}

    status = await load_balancer.get_worker_status()
    if status["healthy_workers"] == 0:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "No healthy workers",
                "details": status,
            },
        )

    return {"status": "healthy", "details": status}


@app.get("/status")
async def status():
    """Get detailed status of all workers."""
    if not load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")

    return await load_balancer.get_worker_status()


@app.get("/connections")
async def connections():
    """Get connection pool statistics for monitoring."""
    if not load_balancer:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")

    return await load_balancer.get_connection_stats()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Load Balancer for ML Workers")
    parser.add_argument(
        "--workers",
        nargs="+",
        required=True,
        help="List of worker URLs (e.g., http://localhost:8001 http://localhost:8002)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=10,
        help="Maximum number of concurrent requests per worker (default: 10)",
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=-1,
        help="Maximum size of the request queue (default: -1, unlimited)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    global load_balancer

    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Initialize load balancer
    load_balancer = LoadBalancer(args.workers, args.max_requests, args.max_queue_size)

    logger.info(f"Starting load balancer with {len(args.workers)} workers")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Max requests per worker: {args.max_requests}")
    logger.info(f"Max queue size: {args.max_queue_size}")

    # Start server
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


async def create_load_balancer(
    worker_urls: List[str], max_requests_per_worker: int, max_queue_size: int = -1
) -> LoadBalancer:
    """Create and initialize a load balancer for standalone use."""
    lb = LoadBalancer(worker_urls, max_requests_per_worker, max_queue_size)
    await lb.check_all_workers_health()
    await lb.start_queue_processor()
    return lb


if __name__ == "__main__":
    main()
