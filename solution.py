from player import Player
from board import HexBoard
import random
import time
import math
import heapq
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
#  1. UNION-FIND
# ─────────────────────────────────────────────────────────────────────────────

class IncrementalUF:
    __slots__ = ['p', 'h']

    def __init__(self, n):
        self.p = list(range(n))
        self.h = [0] * n

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return False
        if   self.h[pa] < self.h[pb]: self.p[pa] = pb
        elif self.h[pa] > self.h[pb]: self.p[pb] = pa
        else:                          self.p[pb] = pa; self.h[pa] += 1
        return True

    def clone(self):
        uf   = IncrementalUF(len(self.p))
        uf.p = self.p[:]
        uf.h = self.h[:]
        return uf


# ─────────────────────────────────────────────────────────────────────────────
#  2. NODO MCTS — RAVE/AMAF + PUCT combinados
#
#  score(parent, child) =
#      (1-b)*Q_mcts  +  b*Q_amaf  +  C*P*sqrt(N_parent) / (1+N_child)
#
#  b  = sqrt( RAVE_K / (3*N_child + RAVE_K) )   -> 0 cuando N_child -> inf
#  P  = prior de path-connection asignado al crear el nodo (fijo)
# ─────────────────────────────────────────────────────────────────────────────

RAVE_K = 314
C_PUCT = 0.8

class MCTSNode:
    __slots__ = ['move', 'player', 'parent', 'children',
                 'untried_moves', 'visits', 'wins',
                 'prior', 'amaf_v', 'amaf_w']

    def __init__(self, move=None, player=None, parent=None, prior=1.0):
        self.move          = move
        self.player        = player
        self.parent        = parent
        self.children      = []
        self.untried_moves = []   # (prior_float, move_tuple), pop() = maximo
        self.visits        = 0
        self.wins          = 0.0
        self.prior         = prior
        self.amaf_v        = {}
        self.amaf_w        = {}

    def combined_score(self, child):
        q_mcts = child.wins / child.visits
        av     = self.amaf_v.get(child.move, 0)
        aw     = self.amaf_w.get(child.move, 0)
        q_amaf = (aw / av) if av > 0 else q_mcts
        beta   = math.sqrt(RAVE_K / (3.0 * child.visits + RAVE_K))
        u_puct = C_PUCT * child.prior * math.sqrt(self.visits) / (1.0 + child.visits)
        return (1.0 - beta) * q_mcts + beta * q_amaf + u_puct


# ─────────────────────────────────────────────────────────────────────────────
#  3. SMART PLAYER
# ─────────────────────────────────────────────────────────────────────────────

class SmartPlayer(Player):

    def __init__(self, player_id):
        super().__init__(player_id)
        self._DIRS_EVEN  = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0),  (1, 1)]
        self._DIRS_ODD = [(-1,-1), (-1, 0), (0, -1), (0, 1), (1,-1),  (1, 0)]
        self._nbrs_cache      = {}
        self._nbrs_cache_size = -1

    # ── Cache de vecindades ──────────────────────────────────────────────────

    def _build_nbrs_cache(self, size):
        cache = {}
        for r in range(size):
            dirs = self._DIRS_EVEN if r % 2 == 0 else self._DIRS_ODD
            for c in range(size):
                cache[(r, c)] = [
                    (r + dr, c + dc)
                    for dr, dc in dirs
                    if 0 <= r + dr < size and 0 <= c + dc < size
                ]
        self._nbrs_cache      = cache
        self._nbrs_cache_size = size

    def _get_nbrs(self, r, c):
        return self._nbrs_cache[(r, c)]

    # ── play() ───────────────────────────────────────────────────────────────

    def play(self, board: HexBoard) -> tuple:
        start = time.time()
        size  = board.size
        pid   = self.player_id
        opp   = 3 - pid
        N2    = size * size
        VL, VR, VT, VB = N2, N2+1, N2+2, N2+3

        if size != self._nbrs_cache_size:
            self._build_nbrs_cache(size)

        state = [row[:] for row in board.board]

        # ── UF base ──────────────────────────────────────────────────────────
        uf_base     = IncrementalUF(N2 + 4)
        valid_empty = []
        for r in range(size):
            for c in range(size):
                v = state[r][c]
                if v:
                    self._add_to_uf(uf_base, r, c, v, size, VL, VR, VT, VB, state)
                else:
                    valid_empty.append((r, c))

        if not valid_empty:
            return (0, 0)

        # ── Prioridad 1: ganar o bloquear en 1-ply ───────────────────────────
        win = self._one_move_check(valid_empty, uf_base, state, pid, size, VL, VR, VT, VB)
        if win:  return win
        blk = self._one_move_check(valid_empty, uf_base, state, opp, size, VL, VR, VT, VB)
        if blk:  return blk

        # ── Prioridad 2: analisis VC ──────────────────────────────────────────
        valid_set = set(valid_empty)

        self_or_win, self_carriers, self_forced = \
            self._vc_analyze(state, size, pid, valid_set)
        if self_forced:
            return self_forced

        opp_or_win, opp_carriers, opp_forced = \
            self._vc_analyze(state, size, opp, valid_set)
        if opp_forced:
            return opp_forced

        # ── Priors path-connection + dead cell pruning ─────────────────────────────
        vc_carriers = (self_carriers | opp_carriers) & valid_set

        # Priors para pid (ofensa propia + defensa vs opp)
        priors_pid, live_empty = self._compute_priors(
            state, size, pid, opp, valid_empty, vc_carriers)

        # Priors para opp (ofensa opp + defensa vs pid)
        # Reutilizamos live_empty del pid (mismas celdas vivas)
        priors_opp, _ = self._compute_priors(
            state, size, opp, pid, live_empty if live_empty else valid_empty, vc_carriers)

        if not live_empty:
            live_empty = valid_empty

        # Tabla indexada por player_id para lookup O(1) en el rollout
        priors_by_player = {pid: priors_pid, opp: priors_opp}

        max_prior_pid = max(priors_pid.values()) if priors_pid else 1.0
        max_prior_opp = max(priors_opp.values()) if priors_opp else 1.0

        # ── Raiz MCTS — solo celdas vivas ────────────────────────────────────
        root = MCTSNode()
        root.untried_moves = sorted((priors_pid.get(m, 0.0), m) for m in live_empty)

        t_overhead = time.time() - start   # todo lo anterior al bucle

        # ── Bucle MCTS ────────────────────────────────────────────────────────
        iters = 0
        while time.time() - start < 4.4:
            iters += 1
            node   = root
            st_sim = [row[:] for row in state]
            uf_sim = uf_base.clone()
            curr   = pid
            path   = []

            # SELECCION
            while not node.untried_moves and node.children:
                node = max(node.children, key=lambda ch, n=node: n.combined_score(ch))
                r, c = node.move
                st_sim[r][c] = curr
                self._add_to_uf(uf_sim, r, c, curr, size, VL, VR, VT, VB, st_sim)
                path.append(node)
                curr = 3 - curr

            # EXPANSION
            if node.untried_moves:
                prior_val, mv = node.untried_moves.pop()
                r, c = mv
                st_sim[r][c] = curr
                self._add_to_uf(uf_sim, r, c, curr, size, VL, VR, VT, VB, st_sim)

                child = MCTSNode(move=mv, player=curr, parent=node, prior=prior_val)
                child.untried_moves = node.untried_moves[:]
                node.children.append(child)
                node  = child
                path.append(node)
                curr  = 3 - curr

            # ROLLOUT con terminacion temprana via UF incremental
            winner, roll_by_player = self._rollout(
                uf_sim, st_sim, size, curr, VL, VR, VT, VB,
                priors_by_player=priors_by_player,
                max_prior_pid=max_prior_pid,
                max_prior_opp=max_prior_opp)

            p1_moves = set(roll_by_player[1])
            p2_moves = set(roll_by_player[2])
            
            # RETROPROPAGACIÓN
            for i in range(len(path) - 1, -1, -1):
                nd = path[i]
                if nd.player == 1: p1_moves.add(nd.move)
                else:              p2_moves.add(nd.move)

                parent  = root if i == 0 else path[i - 1]
                to_play = pid  if parent is root else (3 - parent.player)

                amaf_moves = p1_moves if to_play == 1 else p2_moves
                av_d, aw_d = parent.amaf_v, parent.amaf_w
                wf = (winner == to_play)
                for mv in amaf_moves:
                    av_d[mv] = av_d.get(mv, 0) + 1
                    if wf: aw_d[mv] = aw_d.get(mv, 0) + 1

                nd.visits += 1
                if winner == nd.player:
                    nd.wins += 1.0

            root.visits += 1

        best = max(root.children, key=lambda ch: ch.visits) if root.children else None

        # print(
        #     f"[DEBUG] pid={pid} "
        #     f"overhead={t_overhead:.3f}s  "
        #     f"sims={iters}  "
        #     f"live={len(live_empty)}/{len(valid_empty)}  "
        #     f"best={best.move if best else None}  "
        #     f"visits={best.visits if best else 0}  "
        #     f"winrate={best.wins/best.visits:.2f}" if best else ""
        # )

        if not root.children:
            return live_empty[0]
        return best.move

    # =========================================================================
    #  VC SOLVER
    # =========================================================================

    def _find_bridge_vcs(self, board, size, player):
        bridges = {}
        for r1 in range(size):
            for c1 in range(size):
                if board[r1][c1] != player:
                    continue
                nbrs1 = set(self._get_nbrs(r1, c1))
                for nr, nc in list(nbrs1):
                    for r2, c2 in self._get_nbrs(nr, nc):
                        if board[r2][c2] != player or (r2, c2) == (r1, c1):
                            continue
                        key = ((r1,c1),(r2,c2)) if (r1,c1) < (r2,c2) \
                              else ((r2,c2),(r1,c1))
                        if key in bridges:
                            continue
                        nbrs2  = set(self._get_nbrs(r2, c2))
                        shared = [(r,c) for r,c in nbrs1 & nbrs2 if board[r][c] == 0]
                        if len(shared) == 2:
                            bridges[key] = frozenset(shared)
        return [(a, b, c) for (a, b), c in bridges.items()]

    def _build_group_map(self, board, size, player):
        cell_to_gid  = {}
        gid_to_cells = {}
        gid = 0
        for r in range(size):
            for c in range(size):
                if board[r][c] == player and (r, c) not in cell_to_gid:
                    cells = set()
                    q     = deque([(r, c)])
                    while q:
                        cr, cc = q.popleft()
                        if (cr, cc) in cell_to_gid:
                            continue
                        cell_to_gid[(cr, cc)] = gid
                        cells.add((cr, cc))
                        for nr, nc in self._get_nbrs(cr, cc):
                            if board[nr][nc] == player and (nr, nc) not in cell_to_gid:
                                q.append((nr, nc))
                    gid_to_cells[gid] = cells
                    gid += 1
        return cell_to_gid, gid_to_cells

    def _build_vc_graph(self, board, size, player, bridges, cell_to_gid, gid_to_cells):
        ng     = len(gid_to_cells)
        SOURCE = ng
        DEST   = ng + 1
        adj = {i: [] for i in range(ng)}
        adj[SOURCE] = []
        adj[DEST]   = []

        for gid, cells in gid_to_cells.items():
            ts = td = False
            for r, c in cells:
                if player == 1:
                    if c == 0:        ts = True
                    if c == size - 1: td = True
                else:
                    if r == 0:        ts = True
                    if r == size - 1: td = True
            if ts:
                adj[SOURCE].append((gid, frozenset()))
                adj[gid].append((SOURCE, frozenset()))
            if td:
                adj[DEST].append((gid, frozenset()))
                adj[gid].append((DEST, frozenset()))

        for (r1,c1), (r2,c2), carrier in bridges:
            g1 = cell_to_gid.get((r1, c1))
            g2 = cell_to_gid.get((r2, c2))
            if g1 is not None and g2 is not None and g1 != g2:
                adj[g1].append((g2, carrier))
                adj[g2].append((g1, carrier))

        return adj, SOURCE, DEST

    def _vc_bfs_carriers(self, adj, source, dest):
        if source not in adj:
            return False, set()
        visited = {source: frozenset()}
        queue   = deque([(source, frozenset())])
        while queue:
            node, used = queue.popleft()
            if node == dest:
                return True, set(used)
            for nbr, carrier in adj.get(node, []):
                new_used = used | carrier
                if nbr not in visited or len(visited[nbr]) > len(new_used):
                    visited[nbr] = new_used
                    queue.append((nbr, new_used))
        return False, set()

    def _vc_or_rule_bfs(self, adj, source, dest):
        MAX_STATES = 3000
        visited    = set()
        queue      = deque([(source, frozenset())])
        states     = 0
        while queue and states < MAX_STATES:
            node, used = queue.popleft()
            states += 1
            if node == dest:
                return True, set(used)
            key = (node, used)
            if key in visited:
                continue
            visited.add(key)
            for nbr, carrier in adj.get(node, []):
                if not (carrier & used):
                    queue.append((nbr, used | carrier))
        return False, None

    def _vc_critical_carrier(self, adj, source, dest, valid_set):
        found, base_carriers = self._vc_bfs_carriers(adj, source, dest)
        if not found:
            return None
        cands = [c for c in base_carriers if c in valid_set]
        if not cands:
            return None
        critical = []
        for cell in cands:
            adj_pruned = {
                node: [(nbr, car) for nbr, car in edges if cell not in car]
                for node, edges in adj.items()
            }
            still_ok, _ = self._vc_bfs_carriers(adj_pruned, source, dest)
            if not still_ok:
                critical.append(cell)
        return critical[0] if len(critical) == 1 else None

    def _vc_analyze(self, board, size, player, valid_set):
        bridges = self._find_bridge_vcs(board, size, player)
        cell_to_gid, gid_to_cells = self._build_group_map(board, size, player)
        if not gid_to_cells:
            return False, set(), None

        adj, SOURCE, DEST = self._build_vc_graph(
            board, size, player, bridges, cell_to_gid, gid_to_cells)

        found, carriers = self._vc_bfs_carriers(adj, SOURCE, DEST)
        if not found:
            return False, set(), None

        or_win, _ = self._vc_or_rule_bfs(adj, SOURCE, DEST)

        forced = None
        if not or_win and len(carriers) <= 6:
            forced = self._vc_critical_carrier(adj, SOURCE, DEST, valid_set)

        return or_win, carriers & valid_set, forced

    def _dijkstra(self, board, size, player, forward):
        """Distancia minima de jugadas. propia=0, vacia=1, rival=INF."""
        opp  = 3 - player
        INF  = float('inf')
        dist = [[INF] * size for _ in range(size)]
        heap = []

        if player == 1:
            seeds = [(r, 0 if forward else size - 1) for r in range(size)]
        else:
            seeds = [(0 if forward else size - 1, c) for c in range(size)]

        for r, c in seeds:
            if board[r][c] == opp:
                continue
            d = 0 if board[r][c] == player else 1
            if d < dist[r][c]:
                dist[r][c] = d
                heapq.heappush(heap, (d, r, c))

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue
            for nr, nc in self._get_nbrs(r, c):
                if board[nr][nc] == opp:
                    continue
                nd = d + (0 if board[nr][nc] == player else 1)
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

        return dist

    def _path_connection(self, board, size, player, forward):
        LAMBDA    = 0.7
        EXP_EMPTY = math.exp(-LAMBDA)   # factor para costo 1 (celda vacia)

        opp  = 3 - player
        INF  = float('inf')

        dist = self._dijkstra(board, size, player, forward)
        connection = [[0.0] * size for _ in range(size)]

        # Semillas: cada celda del borde fuente recibe flujo inicial
        if player == 1:
            seeds = [(r, 0 if forward else size - 1) for r in range(size)]
        else:
            seeds = [(0 if forward else size - 1, c) for c in range(size)]

        for r, c in seeds:
            if board[r][c] != opp and dist[r][c] < INF:
                connection[r][c] = 1.0 if board[r][c] == player else EXP_EMPTY

        # DP topologico: ordenar por distancia creciente
        ordered = sorted(
            ((dist[r][c], r, c)
             for r in range(size)
             for c in range(size)
             if board[r][c] != opp and dist[r][c] < INF),
            key=lambda x: x[0]
        )

        for d, r, c in ordered:
            f = connection[r][c]
            if f == 0.0:
                continue
            for nr, nc in self._get_nbrs(r, c):
                if board[nr][nc] == opp:
                    continue
                ec = 0 if board[nr][nc] == player else 1
                # Solo propagar por arcos del DAG (distancia estrictamente creciente)
                if dist[nr][nc] == d + ec:
                    connection[nr][nc] += f if ec == 0 else f * EXP_EMPTY

        return connection, dist

    def _compute_priors(self, board, size, pid, opp, empty_cells, vc_carriers=None):
        VC_BOOST = 2.5

        fwd_p, dist_fp = self._path_connection(board, size, pid, forward=True)
        bwd_p, dist_bp = self._path_connection(board, size, pid, forward=False)
        fwd_o, dist_fo = self._path_connection(board, size, opp, forward=True)
        bwd_o, dist_bo = self._path_connection(board, size, opp, forward=False)

        INF  = float('inf')
        raw  = {}
        live = []

        for r, c in empty_cells:
            pid_useful = dist_fp[r][c] < INF and dist_bp[r][c] < INF
            opp_useful = dist_fo[r][c] < INF and dist_bo[r][c] < INF
            if not pid_useful and not opp_useful:
                continue   # celda muerta: no aporta a ningun jugador

            live.append((r, c))

            # Producto fwd*bwd = densidad de caminos que pasan por (r,c)
            sp = fwd_p[r][c] * bwd_p[r][c]
            so = fwd_o[r][c] * bwd_o[r][c]
            v  = sp + so

            if vc_carriers and (r, c) in vc_carriers:
                v *= VC_BOOST

            raw[(r, c)] = v

        total  = sum(raw.values()) or 1.0
        priors = {m: v / total for m, v in raw.items()}
        return priors, live

    def _rollout(self, uf, board, size, player, VL, VR, VT, VB,
                 priors_by_player=None, max_prior_pid=1.0, max_prior_opp=1.0):
        empty = [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

        if not empty:
            winner = 1 if uf.find(VL) == uf.find(VR) else 2
            return winner, {1: set(), 2: set()}

        # Pre-ordenar por prior de cada jugador (O(n log n), un sort por jugador)
        if priors_by_player:
            def make_order(pd, noise):
                return sorted(
                    empty,
                    key=lambda m: -(pd.get(m, 0.0) + noise * random.random()),
                ) # prior(r,c) + max_prior · Uniform(0,1)
            orders = {
                p: iter(make_order(
                    priors_by_player.get(p, {}),
                    max_prior_pid if p == player else max_prior_opp))
                for p in (1, 2)
            }
            next_cand = {p: next(orders[p], None) for p in (1, 2)}
            taken     = set()
        else:
            random.shuffle(empty)
            orders    = None
            next_cand = None
            taken     = None

        by_player = {1: [], 2: []}
        curr      = player
        winner    = None
        idx       = 0   # puntero para la rama sin priors

        for _ in range(len(empty)):
            # Elegir celda segun prior del jugador actual
            if orders is not None:
                while next_cand[curr] in taken:
                    next_cand[curr] = next(orders[curr], None)
                mv = next_cand[curr]
                if mv is None:
                    break
                taken.add(mv)
                next_cand[curr] = next(orders[curr], None)
            else:
                mv  = empty[idx]
                idx += 1

            r, c = mv
            board[r][c] = curr
            by_player[curr].append(mv)

            # Añadir al UF e inmediatamente chequear victoria — O(alpha)
            self._add_to_uf(uf, r, c, curr, size, VL, VR, VT, VB, board)
            if curr == 1 and uf.find(VL) == uf.find(VR):
                winner = 1
                break
            if curr == 2 and uf.find(VT) == uf.find(VB):
                winner = 2
                break

            curr = 3 - curr

        if winner is None:
            # Fallback (no deberia ocurrir en Hex bien formado)
            winner = 1 if uf.find(VL) == uf.find(VR) else 2

        return winner, {1: set(by_player[1]), 2: set(by_player[2])}

    # =========================================================================
    #  UTILIDADES
    # =========================================================================

    def _add_to_uf(self, uf, r, c, p, size, VL, VR, VT, VB, board):
        cell = r * size + c
        if p == 1:
            if c == 0:        uf.union(cell, VL)
            if c == size - 1: uf.union(cell, VR)
        else:
            if r == 0:        uf.union(cell, VT)
            if r == size - 1: uf.union(cell, VB)
        for nr, nc in self._get_nbrs(r, c):
            if board[nr][nc] == p:
                uf.union(cell, nr * size + nc)

    def _one_move_check(self, valid_empty, uf_base, state, pid, size, VL, VR, VT, VB):
        wl, wr = (VL, VR) if pid == 1 else (VT, VB)
        for r, c in valid_empty:
            uf_check    = uf_base.clone()
            state[r][c] = pid
            self._add_to_uf(uf_check, r, c, pid, size, VL, VR, VT, VB, state)
            state[r][c] = 0
            if uf_check.find(wl) == uf_check.find(wr):
                return (r, c)
        return False