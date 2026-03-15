from player import Player
from board import HexBoard
import random
import time
import math

# ─────────────────────────────────────────────────────────────────────────────
#  1. UNION-FIND INCREMENTAL
# ─────────────────────────────────────────────────────────────────────────────

class IncrementalUF:
    __slots__ = ['p', 'h']
    def __init__(self, n):
        self.p = list(range(n))
        self.h = [0]*n
    
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    
    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa != pb:
            if self.h[pa] < self.h[pb]: self.p[pa] = pb
            elif self.h[pa] > self.h[pb]: self.p[pb] = pa
            else: self.p[pb] = pa; self.h[pa] += 1
            return True
        return False
    
    def clone(self):
        uf = IncrementalUF(len(self.p))
        uf.p = self.p[:]
        uf.h = self.h[:]
        return uf
    

# ─────────────────────────────────────────────────────────────────────────────
#  1. Nodo MCTS (RAVE/AMAF)
# ─────────────────────────────────────────────────────────────────────────────

RAVE_K = 314
UCT_C = 0.1

class MCTSNode:
    __slots__=['move', 'player', 'parent', 'children', 'untried_moves', 'visits', 'wins', 'amaf_v', 'amaf_w']
    def __init__(self, move=None, player=None, parent=None):
        self.move = move
        self.player = player
        self.parent = parent
        self.children = []
        self.untried_moves = []
        self.visits = 0
        self.wins = 0
        self.amaf_v = {}
        self.amaf_w = {}

    def rave_score(self, child):
        mv = child.move
        av = self.amaf_v.get(mv, 0)
        aw = self.amaf_w.get(mv, 0)

        beta = math.sqrt(RAVE_K / (3.0 * child.visits + RAVE_K))
        uct = UCT_C * math.sqrt(math.log(self.visits)/child.visits)

        score_mcts = child.wins / child.visits
        score_amaf = aw / av if av > 0 else 0
        
        return (1 - beta) * score_mcts + beta * score_amaf + uct
        
# ─────────────────────────────────────────────────────────────────────────────
#  3. SMART PLAYER — MCTS + RAVE + UnionFind 
# ─────────────────────────────────────────────────────────────────────────────
class SmartPlayer(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
        
        self._DIRS_ODD = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
        self._DIRS_EVEN  = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]

    def play(self, board: HexBoard) -> tuple:
        start = time.time()
        size  = board.size
        pid   = self.player_id
        opp   = 3 - pid
        N2    = size * size
        VL, VR, VT, VB = N2, N2+1, N2+2, N2+3

        state = [row[:] for row in board.board]

        uf_base = IncrementalUF(N2 + 4)
        valid_empty = []
        for r in range(size):
            for c in range(size):
                if state[r][c] != 0:
                    self._add_to_uf(uf_base, r, c, state[r][c], size, VL, VR, VT, VB, state)
                else: valid_empty.append((r,c))

        if not valid_empty:
            return (0, 0)

        victory = self._victory_check(valid_empty, uf_base, state, pid, size, VL, VR, VT, VB)
        if victory: return victory

        threat = self._victory_check(valid_empty, uf_base, state, opp, size, VL, VR, VT, VB)
        if threat: return threat

        root = MCTSNode()
        root.untried_moves = valid_empty[:]

        while time.time() - start < 4.7:
            node   = root
            st_sim = [row[:] for row in state]
            uf_sim = uf_base.clone()
            curr   = pid
            
            path = []

            # SELECCIÓN
            while not node.untried_moves and node.children:
                node = max(node.children, key=lambda c: node.rave_score(c))
                r, c = node.move
                st_sim[r][c] = curr
                self._add_to_uf(uf_sim, r, c, curr, size, VL, VR, VT, VB, st_sim)
                path.append(node)
                curr = 3 - curr

            # EXPANSIÓN
            if node.untried_moves:
                mv = None
                
                # Intentamos forzar la defensa de un puente basado en la jugada del nodo padre
                if node.move:
                    resp = self._detect_survey_bridge(size, st_sim, curr, node.move[0], node.move[1])
                    if resp and resp in node.untried_moves:
                        mv = resp
                        node.untried_moves.remove(mv)
                
                # Si no hubo ataque a un puente, elegimos uno al azar como siempre
                if not mv:
                    mv = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
                
                st_sim[mv[0]][mv[1]] = curr
                self._add_to_uf(uf_sim, mv[0], mv[1], curr, size, VL, VR, VT, VB, st_sim)
                
                child = MCTSNode(move=mv, player=curr, parent=node)
                # Obtenemos los vacíos para el nuevo hijo
                child.untried_moves = [(r, c) for r in range(size) for c in range(size) if st_sim[r][c] == 0]
                node.children.append(child)
                node = child
                path.append(node)
                curr = 3 - curr

            # SIMULACIÓN (Tablero completo para optimización de Unión/Búsqueda)
            winner, roll_moves = self._rollout_pure(uf_sim, st_sim, size, curr, VL, VR, VT, VB)

            # RETROPROPAGACIÓN (Integración AMAF Corregida)
            p1_subtree_moves = set()
            p2_subtree_moves = set()
            
            # Recopilamos las jugadas del rollout
            for p, m in reversed(roll_moves):
                if p == 1: p1_subtree_moves.add(m)
                else: p2_subtree_moves.add(m)
                
            # Retropropagamos acumulando el historial en el subárbol hacia arriba
            for i in range(len(path) - 1, -1, -1):
                nd = path[i]
                
                if nd.player == 1: p1_subtree_moves.add(nd.move)
                else: p2_subtree_moves.add(nd.move)
                
                parent = path[i-1] if i > 0 else root
                to_play_at_parent = pid if parent is root else 3 - parent.player
                
                moves_to_update = p1_subtree_moves if to_play_at_parent == 1 else p2_subtree_moves
                for mv in moves_to_update:
                    parent.amaf_v[mv] = parent.amaf_v.get(mv, 0) + 1
                    if winner == to_play_at_parent:
                        parent.amaf_w[mv] = parent.amaf_w.get(mv, 0) + 1
                
                nd.visits += 1
                if winner == nd.player:
                    nd.wins += 1

            root.visits += 1

        if root.children:
            return max(root.children, key=lambda c: c.visits).move
        return valid_empty[0]

    def _get_nbrs(self, r, c, size):
        ds = self._DIRS_EVEN if r % 2 == 0 else self._DIRS_ODD

        return [(r + dr, c +  dc) for dr, dc in ds if 0 <= r + dr < size and 0 <= c + dc < size] 
    
    def _add_to_uf(self, uf, r, c, p, size, VL, VR, VT, VB, board):
        cell = r * size + c

        if p == 1:
            if c == 0:
                uf.union(cell, VL)
            if c == size -1:
                uf.union(cell, VR)
        else: 
            if r == 0:
                uf.union(cell, VT)
            if r == size -1:
                uf.union(cell, VB)
        
        for nr, nc in self._get_nbrs(r, c, size):
            if board[nr][nc] == p:
                uf.union(cell, nr * size + nc)
    
    def _victory_check(self, valid_empty, uf_base, state, pid, size, VL, VR, VT, VB):
        for r, c in valid_empty:
            uf_check = uf_base.clone()
            state[r][c] = pid
            self._add_to_uf(uf_check, r, c, pid, size, VL, VR, VT, VB, state)
            state[r][c] = 0
            if (pid == 1 and uf_check.find(VL) == uf_check.find(VR)) or ((pid == 2 and uf_check.find(VT) == uf_check.find(VB))):
                return (r,c)
        return False
    
    def _rollout_pure(self, uf, board, size, player, VL, VR, VT, VB):
        empty_list = [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]
        random.shuffle(empty_list)
        empty_set = set(empty_list) # Para remociones y búsquedas rápidas O(1)
        
        roll_moves = []
        p1_roll = []
        
        last_move = None
        empty_iter = iter(empty_list)
        
        # Simulamos velozmente hasta asfixiar el tablero 
        while empty_set:
            mv = None
            
            # 1. Chequeo de puente: ¿El movimiento anterior amenazó al jugador actual?
            if last_move is not None:
                lr, lc = last_move # Desempaquetamos explícitamente
                resp = self._detect_survey_bridge(size, board, player, lr, lc)
                if resp and resp in empty_set:
                    mv = resp # Movimiento forzado encontrado
            
            # 2. Si no hay amenaza de puente, sacamos el siguiente aleatorio
            if not mv:
                mv = next(empty_iter)
                while mv not in empty_set: # Saltamos los que ya jugamos forzadamente
                    mv = next(empty_iter)
                    
            empty_set.remove(mv)
            r, c = mv
            
            board[r][c] = player
            roll_moves.append((player, mv))
            if player == 1:
                p1_roll.append(mv)
                
            last_move = mv
            player = 3 - player
            
        # Unificamos solo para el jugador 1 al finalizar
        for r, c in p1_roll:
            self._add_to_uf(uf, r, c, 1, size, VL, VR, VT, VB, board)
            
        if uf.find(VL) == uf.find(VR):
            winner = 1
        else:
            winner = 2
            
        return winner, roll_moves
    
    def _valid_move(self, r, c, size):
        return 0 <= r < size and 0 <= c < size
    
    def _cell_to_filling(self, empty_cells, size, board):
        pass

    def _detect_survey_bridge(self, size, board, curr_p, r, c):
        # r,c es un movimiento en una de las dos conexiones de uno de los tipos de puentes
        if self._valid_move(r-1, c+1, size) and self._valid_move(r+1, c, size) and board[r-1][c+1] == curr_p and board[r+1][c] == curr_p and board[r][c+1] == 0:
            return (r, c+1)
        if self._valid_move(r-1, c, size) and self._valid_move(r+1, c-1, size) and board[r-1][c] == curr_p and board[r+1][c-1] == curr_p and board[r][c-1] == 0:
            return (r, c-1)
        if self._valid_move(r, c-1, size) and self._valid_move(r+1, c, size) and board[r][c-1] == curr_p and board[r+1][c] == curr_p and board[r+1][c-1] == 0:
            return (r+1, c-1)
        if self._valid_move(r-1, c, size) and self._valid_move(r, c+1, size) and board[r-1][c] == curr_p and board[r][c+1] == curr_p and board[r-1][c+1] == 0:
            return (r-1, c+1)
        if self._valid_move(r, c+1, size) and self._valid_move(r+1, c-1, size) and board[r][c+1] == curr_p and board[r+1][c-1] == curr_p and board[r + 1][c] == 0:
            return (r + 1, c)
        if self._valid_move(r-1, c+1, size) and self._valid_move(r, c-1, size) and board[r-1][c+1] == curr_p and board[r][c-1] == curr_p and board[r-1][c] == 0:
            return (r-1, c)
        return False