from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv

import os

load_dotenv()


# Constants
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI()


class DeleteUserRequest(BaseModel):
    user_id: str  # UUID of the user


@app.post("/delete-user")
async def delete_user(req: DeleteUserRequest):
    try:
        # Call Supabase Admin API to delete the user
        response = supabase.auth.admin.delete_user(req.user_id)

        if response is None:
            raise HTTPException(status_code=500, detail="Failed to delete user")

        return {"success": True, "message": "User deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
