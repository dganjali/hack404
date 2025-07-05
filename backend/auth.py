from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from database import User, create_user, get_user_by_email, update_user_login

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return payload
    except JWTError:
        return None

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate a user with email and password"""
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

async def register_user(email: str, password: str, name: str = "") -> dict:
    """Register a new user"""
    # Check if user already exists
    existing_user = await get_user_by_email(email)
    if existing_user:
        return {"success": False, "message": "User already exists"}
    
    # Create new user
    hashed_password = hash_password(password)
    user = User(email=email, password_hash=hashed_password, name=name)
    
    success = await create_user(user)
    if success:
        return {"success": True, "message": "User created successfully"}
    else:
        return {"success": False, "message": "Failed to create user"}

async def login_user(email: str, password: str) -> dict:
    """Login a user and return access token"""
    user = await authenticate_user(email, password)
    if not user:
        return {"success": False, "message": "Invalid credentials"}
    
    # Update last login time
    await update_user_login(email)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "role": user.role, "name": user.name},
        expires_delta=access_token_expires
    )
    
    return {
        "success": True,
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "email": user.email,
            "name": user.name,
            "role": user.role
        }
    }

async def get_current_user(token: str) -> Optional[User]:
    """Get current user from token"""
    payload = verify_token(token)
    if payload is None:
        return None
    
    email: str = payload.get("sub")
    if email is None:
        return None
    
    user = await get_user_by_email(email)
    return user

def get_user_from_token(token: str) -> Optional[dict]:
    """Get user info from token without database lookup"""
    payload = verify_token(token)
    if payload is None:
        return None
    
    return {
        "email": payload.get("sub"),
        "role": payload.get("role"),
        "name": payload.get("name")
    } 