import hashlib

def getSEC_Signature(arr):
    combined_string = ''.join(arr)
    hash_value = hashlib.sha256(combined_string.encode()).hexdigest()
    return hash_value

# Usage:
# strings = ["hello", "world", "python", "SHA-256"]
# print(getSEC_Signature(strings))