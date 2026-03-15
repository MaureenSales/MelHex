from __future__ import annotations


class Player:
    """Clase base mínima según la orientación del proyecto."""

    def __init__(self, player_id: int):
        if player_id not in (1, 2):
            raise ValueError("player_id debe ser 1 o 2")
        self.player_id = player_id

    def play(self, board):
        raise NotImplementedError("¡Implementa este método!")
