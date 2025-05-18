import hashlib
raw_code = '123'
hashed_code = hashlib.sha256(raw_code.encode()).hexdigest()
print("hashed_code",hashed_code)