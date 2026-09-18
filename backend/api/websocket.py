import json
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts real-time events."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] Client connected. Total active clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket] Client disconnected. Remaining clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast JSON message to all connected clients."""
        if not self.active_connections:
            return

        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)

# Global singleton connection manager
ws_manager = ConnectionManager()
