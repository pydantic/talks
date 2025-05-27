import os

import logfire
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from pydantic_ai.agent import Agent
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.agent import build_agent, answer_question, BotResponse

logfire.configure(
    service_name='api',
    environment='staging'
)

# FastAPI application setup
app = FastAPI(title="Math, Database and PydanticAI API")
logfire.instrument_fastapi(app)


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
logfire.instrument_sqlalchemy(engine)


# Database model
class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)


Base.metadata.create_all(bind=engine)


# Pydantic models
class ItemCreate(BaseModel):
    name: str
    description: str


class ItemResponse(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class AgentQuery(BaseModel):
    question: str


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Dependency to get agent instance
def get_agent() -> Agent[None, BotResponse]:
    return build_agent()


# Endpoint 1: Division
@app.get("/divide/{numerator}/{denominator}")
async def divide(numerator: float, denominator: float):
    """
    Divides the numerator by the denominator and returns the result.
    """
    result = numerator / denominator
    return {"result": result}


# Endpoint 2: Fibonacci
@app.get("/fibonacci/{n}")
async def fibonacci(n: int):
    """
    Calculates the nth number in the Fibonacci sequence.
    Raises an HTTPException if n is negative.
    """
    if n < 0:
        raise HTTPException(status_code=400, detail="Input must be a non-negative integer")

    if n <= 1:
        return {"result": n}

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return {"result": b}


# Endpoint 3: Database Query
@app.post("/items/", response_model=ItemResponse)
async def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    """
    Creates a new item in the database.
    """
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@app.get("/items/", response_model=list[ItemResponse])
async def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieves items from the database with pagination.
    """
    items = db.query(Item).offset(skip).limit(limit).all()
    return items


@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int, db: Session = Depends(get_db)):
    """
    Retrieves a specific item by ID.
    Raises an HTTPException if the item is not found.
    """
    item = db.query(Item).filter(Item.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.post("/agent/query", response_model=BotResponse)
async def query_agent(query: AgentQuery, agent: Agent[None, BotResponse] = Depends(get_agent)):
    """
    Queries the PydanticAI agent with a user question and returns the response.
    """
    logfire.info(f"Querying agent with question: {query.question}")
    response = await answer_question(agent, query.question)
    return response


if __name__ == '__main__':  # Fixed double asterisks
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Added this line to complete the if block