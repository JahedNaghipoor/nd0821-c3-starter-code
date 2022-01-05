import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_main_route_OK():
    route = client.get("/")
    assert route.status_code == 200
    
def test_main_route_message():
    route = client.get("/")
    assert route.json() == "Greetings!!!"