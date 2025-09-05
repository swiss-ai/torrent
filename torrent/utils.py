import random
import string


TMP_PATH = "/users/nathanrchn/torrent_NAFHueoW"


def nanoid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters, k=length))
