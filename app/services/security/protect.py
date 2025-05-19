from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import base64

def generate_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt(plain_text: str, key: bytes) -> str:
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ct = encryptor.update(plain_text.encode()) + encryptor.finalize()
    return base64.b64encode(iv + ct).decode('utf-8')  # Return as base64 string

def decrypt(encrypted_b64: str, key: bytes) -> str:
    encrypted = base64.b64decode(encrypted_b64.encode('utf-8'))
    iv = encrypted[:16]
    ct = encrypted[16:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return (decryptor.update(ct) + decryptor.finalize()).decode()

# Usage:
# password = os.getenv("SEC_PROTECT", "default-fallback")
# # password = "mypassword"
# salt = os.urandom(16)
# key = generate_key(password, salt)

# message = "OK, understadn what you mean. So how ddo you think?"
# encrypted_msg = encrypt(message, key)

# print("Encrypted (bytes):", encrypted_msg)

# decrypted_msg = decrypt(encrypted_msg, key)
# print("Decrypted:", decrypted_msg)