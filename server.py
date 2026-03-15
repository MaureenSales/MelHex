from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from board import HexBoard
from custom_solution import SmartPlayer

app = FastAPI(title="HEX SmartPlayer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Modelos ────────────────────────────────────────────
class PlayRequest(BaseModel):
    board: List[List[int]]   
    player_id: int            
    size: int                 

class PlayResponse(BaseModel):
    row: int
    col: int

class CheckWinnerRequest(BaseModel):
    board: List[List[int]]
    size: int

class CheckWinnerResponse(BaseModel):
    winner: int

# ── Endpoints ──────────────────────────────────────────
@app.post("/play", response_model=PlayResponse)
def play(req: PlayRequest):
    if req.player_id not in (1, 2):
        raise HTTPException(400, "player_id debe ser 1 o 2")
    if len(req.board) != req.size:
        raise HTTPException(400, "Tamaño del tablero inconsistente")

    hex_board = HexBoard(req.size)
    hex_board.board = [row[:] for row in req.board]

    player = SmartPlayer(req.player_id)
    move = player.play(hex_board)

    return PlayResponse(row=move[0], col=move[1])

@app.get("/health")
def health():
    return {"status": "ok", "message": "SmartPlayer listo"}