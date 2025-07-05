from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Optional
import os

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "needsmatcher"

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def get_database() -> AsyncIOMotorClient:
    return db.client[DATABASE_NAME]

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(MONGODB_URL)
    print("✅ Connected to MongoDB")
    print(f"📊 Database URL: {MONGODB_URL}")

async def close_mongo_connection():
    if db.client:
        db.client.close()
        print("✅ Disconnected from MongoDB")

# User model
class User:
    def __init__(self, email: str, password_hash: str, name: str = "", role: str = "user"):
        self.email = email
        self.password_hash = password_hash
        self.name = name
        self.role = role
        self.created_at = datetime.utcnow()
        self.last_login = None

    def to_dict(self):
        return {
            "email": self.email,
            "password_hash": self.password_hash,
            "name": self.name,
            "role": self.role,
            "created_at": self.created_at,
            "last_login": self.last_login
        }

    @classmethod
    def from_dict(cls, data: dict):
        user = cls(
            email=data["email"],
            password_hash=data["password_hash"],
            name=data.get("name", ""),
            role=data.get("role", "user")
        )
        user.created_at = data.get("created_at", datetime.utcnow())
        user.last_login = data.get("last_login")
        return user

# Database operations
async def create_user(user: User) -> bool:
    """Create a new user in the database"""
    try:
        database = await get_database()
        users_collection = database.users
        
        # Check if user already exists
        existing_user = await users_collection.find_one({"email": user.email})
        if existing_user:
            return False
        
        # Insert new user
        await users_collection.insert_one(user.to_dict())
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False

async def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email"""
    try:
        database = await get_database()
        users_collection = database.users
        
        user_data = await users_collection.find_one({"email": email})
        if user_data:
            return User.from_dict(user_data)
        return None
    except Exception as e:
        print(f"Error getting user: {e}")
        return None

async def update_user_login(email: str):
    """Update user's last login time"""
    try:
        database = await get_database()
        users_collection = database.users
        
        await users_collection.update_one(
            {"email": email},
            {"$set": {"last_login": datetime.utcnow()}}
        )
    except Exception as e:
        print(f"Error updating user login: {e}")

async def get_all_users():
    """Get all users (for admin purposes)"""
    try:
        database = await get_database()
        users_collection = database.users
        
        users = []
        async for user_data in users_collection.find():
            users.append(User.from_dict(user_data))
        return users
    except Exception as e:
        print(f"Error getting all users: {e}")
        return []

# Initialize database with sample data
async def init_database():
    """Initialize database with sample data if empty"""
    try:
        database = await get_database()
        
        # Check if users collection exists and has data
        users_count = await database.users.count_documents({})
        
        # Check if we should create sample user
        create_sample = os.getenv("CREATE_SAMPLE_USER", "true").lower() == "true"
        sample_email = os.getenv("SAMPLE_USER_EMAIL", "admin@needsmatcher.com")
        sample_password = os.getenv("SAMPLE_USER_PASSWORD", "admin123")
        
        if users_count == 0 and create_sample:
            # Create sample admin user
            from auth import hash_password
            admin_user = User(
                email=sample_email,
                password_hash=hash_password(sample_password),
                name="Admin User",
                role="admin"
            )
            await create_user(admin_user)
            print(f"✅ Created sample admin user: {sample_email} / {sample_password}")
        
        print(f"✅ Database initialized with {users_count} users")
    except Exception as e:
        print(f"Error initializing database: {e}") 