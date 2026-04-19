import secrets

def generate_secret_key():
    """Generate a secret key using the secrets module."""
    return secrets.token_urlsafe(16)