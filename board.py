from __future__ import annotations
from typing import List, Tuple
from collections import deque


class HexBoard:
    """
    Tablero hexagonal para el juego HEX usando sistema even-r layout.
    
    - Jugador 1 (id=1): Conecta izquierda (columna 0) con derecha (columna size-1)
    - Jugador 2 (id=2): Conecta arriba (fila 0) con abajo (fila size-1)
    """
    
    # Direcciones de adyacencia para sistema even-r layout
    # Filas PARES (0, 2, 4...): vecinos hacia la izquierda
    _DIRS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    # Filas IMPARES (1, 3, 5...): vecinos hacia la derecha
    _DIRS_ODD = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    
    def __init__(self, size: int):
        """
        Inicializa un tablero HEX de tamaño size x size.
        
        Args:
            size: Tamaño del tablero (N x N)
        """
        if size < 2:
            raise ValueError("El tamaño del tablero debe ser al menos 2")
        
        self.size = size
        self.board: List[List[int]] = [[0 for _ in range(size)] for _ in range(size)]
        # 0 = vacío, 1 = Jugador 1, 2 = Jugador 2
    
    def clone(self) -> HexBoard:
        """
        Devuelve una copia profunda del tablero actual.
        
        Returns:
            Nueva instancia de HexBoard con el mismo estado
        """
        new_board = HexBoard(self.size)
        new_board.board = [row[:] for row in self.board]
        return new_board
    
    def place_piece(self, row: int, col: int, player_id: int) -> bool:
        """
        Coloca una ficha en el tablero si la casilla está vacía.
        
        Args:
            row: Fila donde colocar (0-indexed)
            col: Columna donde colocar (0-indexed)
            player_id: ID del jugador (1 o 2)
        
        Returns:
            True si se colocó exitosamente, False si no
        """
        if not self._is_valid_position(row, col):
            return False
        
        if self.board[row][col] != 0:
            return False
        
        if player_id not in (1, 2):
            return False
        
        self.board[row][col] = player_id
        return True
    
    def check_connection(self, player_id: int) -> bool:
        """
        Verifica si el jugador ha conectado sus dos lados opuestos usando BFS.
        
        Jugador 1: debe conectar columna 0 (izquierda) con columna size-1 (derecha)
        Jugador 2: debe conectar fila 0 (arriba) con fila size-1 (abajo)
        
        Args:
            player_id: ID del jugador a verificar (1 o 2)
        
        Returns:
            True si el jugador ha ganado, False si no
        """
        if player_id not in (1, 2):
            return False
        
        if player_id == 1:
            return self._check_horizontal_connection()
        else:
            return self._check_vertical_connection()
    
    def _check_horizontal_connection(self) -> bool:
        """
        Verifica si el Jugador 1 conecta izquierda (col 0) con derecha (col size-1).
        """
        visited = [[False] * self.size for _ in range(self.size)]
        queue = deque()
        
        # Iniciar BFS desde todas las celdas de la columna izquierda (col 0)
        for row in range(self.size):
            if self.board[row][0] == 1:
                queue.append((row, 0))
                visited[row][0] = True
        
        # BFS
        while queue:
            row, col = queue.popleft()
            
            # Si llegamos a la columna derecha, hay conexión
            if col == self.size - 1:
                return True
            
            # Explorar vecinos
            for neighbor_row, neighbor_col in self._get_neighbors(row, col):
                if not visited[neighbor_row][neighbor_col] and self.board[neighbor_row][neighbor_col] == 1:
                    visited[neighbor_row][neighbor_col] = True
                    queue.append((neighbor_row, neighbor_col))
        
        return False
    
    def _check_vertical_connection(self) -> bool:
        """
        Verifica si el Jugador 2 conecta arriba (row 0) con abajo (row size-1).
        """
        visited = [[False] * self.size for _ in range(self.size)]
        queue = deque()
        
        # Iniciar BFS desde todas las celdas de la fila superior (row 0)
        for col in range(self.size):
            if self.board[0][col] == 2:
                queue.append((0, col))
                visited[0][col] = True
        
        # BFS
        while queue:
            row, col = queue.popleft()
            
            # Si llegamos a la fila inferior, hay conexión
            if row == self.size - 1:
                return True
            
            # Explorar vecinos
            for neighbor_row, neighbor_col in self._get_neighbors(row, col):
                if not visited[neighbor_row][neighbor_col] and self.board[neighbor_row][neighbor_col] == 2:
                    visited[neighbor_row][neighbor_col] = True
                    queue.append((neighbor_row, neighbor_col))
        
        return False
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Obtiene las coordenadas de los vecinos válidos para una celda.
        Usa el sistema even-r layout.
        
        Args:
            row: Fila de la celda
            col: Columna de la celda
        
        Returns:
            Lista de tuplas (fila, columna) con vecinos válidos
        """
        # Seleccionar direcciones según si la fila es par o impar
        directions = self._DIRS_EVEN if row % 2 == 0 else self._DIRS_ODD
        
        neighbors = []
        for dr, dc in directions:
            neighbor_row = row + dr
            neighbor_col = col + dc
            
            if self._is_valid_position(neighbor_row, neighbor_col):
                neighbors.append((neighbor_row, neighbor_col))
        
        return neighbors
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """
        Verifica si una posición está dentro del tablero.
        
        Args:
            row: Fila a verificar
            col: Columna a verificar
        
        Returns:
            True si es válida, False si no
        """
        return 0 <= row < self.size and 0 <= col < self.size
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Obtiene todas las celdas vacías del tablero.
        
        Returns:
            Lista de tuplas (fila, columna) con las posiciones vacías
        """
        empty = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == 0:
                    empty.append((row, col))
        return empty
    
    def is_full(self) -> bool:
        """
        Verifica si el tablero está completamente lleno.
        
        Returns:
            True si no hay celdas vacías, False si no
        """
        return len(self.get_empty_cells()) == 0
    
    def __str__(self) -> str:
        """
        Representación en string del tablero para debugging.
        """
        result = []
        for row_idx, row in enumerate(self.board):
            indent = " " * row_idx if row_idx % 2 == 1 else ""
            row_str = indent + " ".join(str(cell) for cell in row)
            result.append(row_str)
        return "\n".join(result)
