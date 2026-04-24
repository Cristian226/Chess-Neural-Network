import math
import time
from array import array
from threading import Event
import chess
import chess.polyglot
import torch

from ai.encoding import encode_board
from ai.model import ChessEvalNet
from ai.training_config import TANH_SCALE
from config import AI_MODEL_PATH


PIECE_VALUE: dict[int, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20_000,
}

# Bitboard attack tables (module-level to avoid per-call attribute lookup)
_PAWN_ATK   = chess.BB_PAWN_ATTACKS
_KNIGHT_ATK = chess.BB_KNIGHT_ATTACKS
_KING_ATK   = chess.BB_KING_ATTACKS
_DIAG_ATK   = chess.BB_DIAG_ATTACKS
_RANK_ATK   = chess.BB_RANK_ATTACKS
_FILE_ATK   = chess.BB_FILE_ATTACKS
_DIAG_MASK  = chess.BB_DIAG_MASKS
_RANK_MASK  = chess.BB_RANK_MASKS
_FILE_MASK  = chess.BB_FILE_MASKS
_BB_SQ      = chess.BB_SQUARES

_SEE_ORDER       = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_SEE_LAZY_MARGIN = 50   # skip full SEE when victim > attacker + this margin

# Search constants
MATE_SCORE          = 100_000
MAX_PLY             = 128
TT_SIZE             = 1 << 20
EVAL_CACHE_SIZE     = 1 << 20
STOP_CHECK_INTERVAL = 4_096
QSEARCH_MAX_DEPTH   = 8

# Draw-conversion tuning (side-to-move perspective, in cp)
WINNING_DRAW_CP  = 250
DRAW_SCORE_WHEN_WINNING = -40
DRAW_SCORE_WHEN_LOSING  = 10
REPETITION_PENALTY_WHEN_WINNING = 80

_TT_DEPTH = 0; _TT_VALUE = 1; _TT_FLAG = 2; _TT_MOVE = 3
EXACT = 0; LOWER = 1; UPPER = 2

# Flat LMR table: index = depth * 64 + move_index
_LMR: list[int] = [0] * (64 * 64)
for _d in range(1, 64):
    for _m in range(1, 64):
        _LMR[_d * 64 + _m] = max(1, int(1.0 + math.log(_d) * math.log(_m) / 2.0))

_LMP: list[int] = [0, 3, 4, 7, 10, 14, 19, 25]

# Move-ordering priority tiers
_TT_SCORE      = 10_000_000
_PROMO_SCORE   =  9_000_000
_GOOD_CAP_BASE =  5_000_000
_KILLER0       =  4_000_000
_KILLER1       =  3_000_000
_COUNTER       =  2_000_000
_BAD_CAP_BASE  = -3_000_000
_UNDERPROM     = -8_000_000

_CH_DIM = 384   # continuation history: (piece_type − 1) * 64 + square


class NeuralNetEngine:
    def __init__(self, depth: int, time_limit_ms: int | None = None) -> None:
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self._cancel_event = Event()
        self._stop_search = False

        ckpt = torch.load(AI_MODEL_PATH, map_location="cpu", weights_only=False)
        self._tanh_scale = float(ckpt.get("tanh_scale", TANH_SCALE))
        model = ChessEvalNet()
        model.load_state_dict(ckpt["model"])
        model.eval()

        dummy = torch.zeros(1, 20, 8, 8)
        self.model = torch.jit.freeze(torch.jit.trace(model, dummy))
        with torch.no_grad():
            for _ in range(5):
                self.model(dummy)

        self._eval_cache:  dict[int, float] = {}
        self.tt: dict[int, tuple] = {}
        self.killer_moves: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
        self.history = array('l', [0] * 4096)
        self.cont_hist = array('l', [0] * (_CH_DIM * _CH_DIM))
        self.countermoves: list[list[list[chess.Move | None]]] = [[[None] * 64 for _ in range(64)] for _ in range(2)]
        self._eval_stack: list[float] = [0.0] * MAX_PLY
        self._eval_valid: list[bool] = [False] * MAX_PLY
        self._deadline_ns: int | None = None
        self.nodes = 0


    def cancel_search(self) -> None:
        self._cancel_event.set()
        self._stop_search = True

    def _time_is_up(self) -> bool:
        return self._deadline_ns is not None and time.perf_counter_ns() >= self._deadline_ns


    def evaluate(self, board: chess.Board, key: int | None = None) -> float:
        if key is None:
            key = chess.polyglot.zobrist_hash(board)
        cached = self._eval_cache.get(key)
        if cached is not None:
            return cached

        with torch.inference_mode():
            pred = self.model(encode_board(board, clone=False).unsqueeze(0)).item()

        x = max(min(pred, 1.0 - 1e-6), -1.0 + 1e-6)
        cp = math.atanh(x) * self._tanh_scale

        self._eval_cache[key] = cp
        if len(self._eval_cache) > EVAL_CACHE_SIZE:
            self._eval_cache.clear()
        return cp

    def evaluate_position(self, board: chess.Board) -> int:
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        cp = self.evaluate(board)
        return int(round(cp if board.turn == chess.WHITE else -cp))

    def _draw_score_from_eval(self, stm_eval_cp: float) -> int:
        if stm_eval_cp > WINNING_DRAW_CP:
            scale = min(120, int((stm_eval_cp - WINNING_DRAW_CP) * 0.1))
            return DRAW_SCORE_WHEN_WINNING - scale
        if stm_eval_cp < -WINNING_DRAW_CP:
            scale = min(50, int((-stm_eval_cp - WINNING_DRAW_CP) * 0.05))
            return DRAW_SCORE_WHEN_LOSING + scale
        return 0

    # Static Exchange Evaluation
    def _see(self, board: chess.Board, move: chess.Move, threshold = 0) -> bool:
        to_sq = move.to_square; from_sq = move.from_square

        if board.is_en_passant(move):
            captured_val = PIECE_VALUE[chess.PAWN]
            ep_sq = board.ep_square + (-8 if board.turn == chess.WHITE else 8)
        else:
            ct = board.piece_type_at(to_sq)
            if ct is None: return threshold <= 0
            captured_val = PIECE_VALUE[ct]; ep_sq = None

        value = captured_val - threshold
        if value < 0: return False
        attacker_pt = board.piece_type_at(from_sq)
        if attacker_pt is None: return True
        value -= PIECE_VALUE[attacker_pt]
        if value >= 0: return True

        occ = board.occupied ^ _BB_SQ[from_sq]
        occ ^= _BB_SQ[ep_sq] if ep_sq is not None else _BB_SQ[to_sq]
        diag_bb = board.bishops | board.queens
        orth_bb = board.rooks   | board.queens
        dm, rm, fm = _DIAG_MASK[to_sq], _RANK_MASK[to_sq], _FILE_MASK[to_sq]
        da, ra, fa = _DIAG_ATK[to_sq],  _RANK_ATK[to_sq],  _FILE_ATK[to_sq]
        attackers = (
            (_PAWN_ATK[chess.WHITE][to_sq] & board.pawns & board.occupied_co[chess.BLACK]) |
            (_PAWN_ATK[chess.BLACK][to_sq] & board.pawns & board.occupied_co[chess.WHITE]) |
            (_KNIGHT_ATK[to_sq] & board.knights) | (_KING_ATK[to_sq] & board.kings) |
            (da[occ & dm] & diag_bb) | (ra[occ & rm] & orth_bb) | (fa[occ & fm] & orth_bb)
        ) & occ

        pieces_mask = board.pieces_mask; occupied_co = board.occupied_co; stm = not board.turn
        while True:
            stm_att = attackers & occupied_co[stm]
            if not stm_att: break
            for pt in _SEE_ORDER:
                bb = pieces_mask(pt, stm) & stm_att
                if bb: break
            else: break
            sq = chess.lsb(bb); occ ^= _BB_SQ[sq]
            if pt in (chess.PAWN, chess.BISHOP, chess.QUEEN): attackers |= da[occ & dm] & diag_bb
            if pt in (chess.ROOK, chess.QUEEN): attackers |= (ra[occ & rm] | fa[occ & fm]) & orth_bb
            attackers &= occ; value = -value - 1 - PIECE_VALUE[pt]; stm = not stm
            if value >= 0:
                if pt == chess.KING and (attackers & occupied_co[stm]): stm = not stm
                break
        return board.turn != stm

    # Move ordering
    def _order_moves(self, board: chess.Board, ply: int, tt_move: chess.Move | None, prev_move: chess.Move | None) -> list[chess.Move]:
        cm = (self.countermoves[not board.turn][prev_move.from_square][prev_move.to_square]
              if prev_move else None)
        k0, k1   = self.killer_moves[ply]
        piece_at = board.piece_type_at
        is_cap = board.is_capture
        history  = self.history
        ch = self.cont_hist

        prev_ch_key = -1
        if prev_move:
            prev_pt = piece_at(prev_move.to_square)
            if prev_pt: prev_ch_key = (prev_pt - 1) * 64 + prev_move.to_square

        scored: list[tuple[int, chess.Move]] = []
        for move in board.legal_moves:
            f, t = move.from_square, move.to_square
            if move == tt_move:
                s = _TT_SCORE
            elif move.promotion:
                s = _PROMO_SCORE if move.promotion == chess.QUEEN else _UNDERPROM
            elif is_cap(move):
                vv = PIECE_VALUE.get(piece_at(t), 0)
                av = PIECE_VALUE.get(piece_at(f), 0)
                mvv = vv * 10 - av
                # Lazy SEE: skip exchange sim for clearly winning captures.
                s = (_GOOD_CAP_BASE + mvv if vv > av + _SEE_LAZY_MARGIN
                     else (_GOOD_CAP_BASE if self._see(board, move, 0) else _BAD_CAP_BASE) + mvv)
            elif move == k0: s = _KILLER0
            elif move == k1: s = _KILLER1
            elif move == cm: s = _COUNTER
            else:
                ch_bonus = 0
                if prev_ch_key >= 0:
                    curr_pt = piece_at(f)
                    if curr_pt: ch_bonus = ch[prev_ch_key * _CH_DIM + (curr_pt - 1) * 64 + t]
                s = history[f * 64 + t] + ch_bonus
            scored.append((s, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # History / killer / countermove updates
    def _update_quiet_stats(self, board: chess.Board, move: chess.Move, depth, ply, prev_move: chess.Move | None, quiets_searched: list[chess.Move]) -> None:
        killers = self.killer_moves[ply]
        if move != killers[0]: killers[1] = killers[0]; killers[0] = move
        if prev_move:
            self.countermoves[not board.turn][prev_move.from_square][prev_move.to_square] = move

        bonus = min(depth * depth, 1200)
        h, ch = self.history, self.cont_hist

        prev_ch_key = -1
        if prev_move:
            prev_pt = board.piece_type_at(prev_move.to_square)
            if prev_pt: prev_ch_key = (prev_pt - 1) * 64 + prev_move.to_square

        def _chk(m: chess.Move) -> int:
            pt = board.piece_type_at(m.from_square)
            return (pt - 1) * 64 + m.to_square if pt else -1

        idx = move.from_square * 64 + move.to_square
        h[idx] = min(h[idx] + bonus, 400_000)
        if prev_ch_key >= 0:
            cki = _chk(move)
            if cki >= 0:
                i = prev_ch_key * _CH_DIM + cki; ch[i] = min(ch[i] + bonus, 400_000)

        for q in quiets_searched:
            qi = q.from_square * 64 + q.to_square; h[qi] = max(h[qi] - bonus, -400_000)
            if prev_ch_key >= 0:
                cki = _chk(q)
                if cki >= 0:
                    i = prev_ch_key * _CH_DIM + cki; ch[i] = max(ch[i] - bonus, -400_000)

    def _age_history(self) -> None:
        h, ch = self.history, self.cont_hist
        for i in range(4096): h[i] >>= 1
        for i in range(_CH_DIM * _CH_DIM):  ch[i] >>= 1

    # Quiescence search
    def _qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int, qs_depth: int = 0) -> int:
        self.nodes += 1
        if self._stop_search or (
            not (self.nodes & (STOP_CHECK_INTERVAL - 1))
            and (self._cancel_event.is_set() or self._time_is_up())
        ):
            self._stop_search = True; return alpha

        if self._time_is_up():
            self._stop_search = True
            return alpha

        in_check = board.is_check()
        if not in_check:
            stand_pat = self.evaluate(board)
            if stand_pat >= beta: return beta
            if stand_pat > alpha: alpha = stand_pat
            if qs_depth >= QSEARCH_MAX_DEPTH: return alpha

        piece_at = board.piece_type_at
        if in_check:
            moves: list[chess.Move] = list(board.legal_moves)
            if not moves: return -MATE_SCORE + ply
        else:
            moves = sorted(board.generate_legal_captures(),
                           key=lambda m: PIECE_VALUE.get(piece_at(m.to_square), 0) * 10
                                         - PIECE_VALUE.get(piece_at(m.from_square), 0),
                           reverse=True)

        for move in moves:
            if not in_check:
                victim = piece_at(move.to_square)
                if stand_pat + PIECE_VALUE.get(victim, 100) + 200 < alpha: continue
                if not self._see(board, move, 0): continue
            board.push(move)
            score = -self._qsearch(board, -beta, -alpha, ply + 1,
                                   qs_depth if in_check else qs_depth + 1)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha

    # Main alpha-beta search, all helpers inlined
    def _search(self, board: chess.Board, depth, alpha, beta, ply, prev_move: chess.Move | None = None, cut_node = False, excluded_move: chess.Move | None = None,) -> tuple[int, chess.Move | None]:
        self.nodes += 1
        if self._stop_search: return alpha, None
        if self._time_is_up():
            self._stop_search = True
            return alpha, None
        if not (self.nodes & (STOP_CHECK_INTERVAL - 1)) and (self._cancel_event.is_set() or self._time_is_up()):
            self._stop_search = True; return alpha, None

        # Draw detection
        if board.is_insufficient_material(): return 0, None
        if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
            return self._draw_score_from_eval(self.evaluate(board)), None
        if board.is_repetition(2):
            rep_eval = self.evaluate(board)
            rep_score = self._draw_score_from_eval(rep_eval)
            if rep_eval > WINNING_DRAW_CP:
                rep_score -= REPETITION_PENALTY_WHEN_WINNING + min(120, int((rep_eval - WINNING_DRAW_CP) * 0.2))
            return rep_score, None

        # Mate-distance pruning
        if -MATE_SCORE + ply > alpha: alpha = -MATE_SCORE + ply
        if  MATE_SCORE - ply - 1 < beta: beta = MATE_SCORE - ply - 1
        if alpha >= beta: return alpha, None
        if depth <= 0: return self._qsearch(board, alpha, beta, ply), None

        root = ply == 0
        pv_node = beta - alpha > 1
        key = chess.polyglot.zobrist_hash(board)

        # TT probe
        entry   = None if excluded_move else self.tt.get(key)
        tt_move: chess.Move | None = entry[_TT_MOVE] if entry else None
        if entry and not root and entry[_TT_DEPTH] >= depth:
            ev, ef = entry[_TT_VALUE], entry[_TT_FLAG]
            if   ef == EXACT: return ev, tt_move
            elif ef == LOWER and ev > alpha: alpha = ev
            elif ef == UPPER and ev < beta:  beta  = ev
            if alpha >= beta: return alpha, tt_move

        # Static eval
        in_check = board.is_check()
        if in_check:
            static_eval = 0.0; eval_valid = False; improving = False
        else:
            static_eval = (float(entry[_TT_VALUE]) if entry and entry[_TT_FLAG] == EXACT
                           else self.evaluate(board, key))
            eval_valid = True
            improving  = (ply >= 2 and self._eval_valid[ply - 2]
                          and static_eval > self._eval_stack[ply - 2])
        winning_side_to_move = eval_valid and static_eval > WINNING_DRAW_CP
        self._eval_stack[ply] = static_eval
        self._eval_valid[ply] = eval_valid
        not_mate = abs(alpha) < MATE_SCORE - 200 and abs(beta) < MATE_SCORE - 200

        # Pre-move pruning
        if not pv_node and not in_check and not_mate and eval_valid and not excluded_move:

            # Reverse futility pruning
            if depth <= 6 and static_eval - (75 * depth - 60 * improving) >= beta:
                return beta, None

            # Razoring
            if depth <= 3 and static_eval + 180 * depth <= alpha:
                return self._qsearch(board, alpha, beta, ply), None

            # Null-move pruning
            if depth >= 2 and static_eval >= beta and ply > 0 and not winning_side_to_move:
                has_pieces = bool(board.occupied_co[board.turn] & ~board.pawns & ~board.kings)
                if has_pieces:
                    R = 3 + depth // 3 + min(3, (int(static_eval) - int(beta)) // 200)
                    board.push(chess.Move.null())
                    nmp, _ = self._search(board, depth - 1 - R, -beta, -beta + 1,
                                          ply + 1, None, not cut_node)
                    board.pop()
                    if -nmp >= beta: return min(-nmp, MATE_SCORE - 200), None

        # Internal iterative reduction
        if depth >= 4 and tt_move is None: depth -= 1

        # Singular extensions
        extension = 0
        if (not root and excluded_move is None and depth >= 7
                and tt_move is not None and entry is not None
                and entry[_TT_DEPTH] >= depth - 3 and entry[_TT_FLAG] != UPPER):
            s_beta = max(entry[_TT_VALUE] - 2 * depth, -MATE_SCORE)
            s_score, _ = self._search(board, (depth - 1) // 2, s_beta - 1, s_beta,
                                      ply, prev_move, cut_node, excluded_move=tt_move)
            if s_score < s_beta: extension = 1
            elif s_beta >= beta: return s_beta, tt_move

        # Move loop
        moves = self._order_moves(board, ply, tt_move, prev_move)
        history = self.history
        best_move: chess.Move | None = None
        best_val = -MATE_SCORE - 1
        orig_alpha = alpha
        quiets_searched: list[chess.Move] = []
        move_count = 0

        for move in moves:
            if move == excluded_move: continue
            if self._stop_search:
                return (best_val if best_val > -MATE_SCORE else alpha), best_move

            is_capture = board.is_capture(move)
            is_quiet = not is_capture and not move.promotion
            move_count += 1

            # Protect advanced pawn pushes from pruning
            is_advanced_pawn = (
                is_quiet
                and board.piece_type_at(move.from_square) == chess.PAWN
                and ((board.turn == chess.WHITE and chess.square_rank(move.to_square) >= 5)
                  or (board.turn == chess.BLACK and chess.square_rank(move.to_square) <= 2))
            )

            # Late-move / futility pruning (quiet moves only)
            if is_quiet and not is_advanced_pawn and best_move is not None and not in_check and not_mate:
                if depth < len(_LMP) and move_count > _LMP[depth]: continue
                if depth <= 6 and eval_valid and static_eval + 60 + 90 * depth <= alpha: continue
                if depth <= 3 and history[move.from_square * 64 + move.to_square] < -120 * depth: continue

            # SEE pruning for losing captures
            if (is_capture and move_count > 1 and best_move is not None
                    and not in_check and not_mate and depth <= 6
                    and not self._see(board, move, -50 * depth)):
                continue

            board.push(move)
            gives_check = board.is_check()
            new_depth   = depth - 1 + gives_check + (extension if move_count == 1 else 0)

            if move_count == 1:
                score, _ = self._search(board, new_depth, -beta, -alpha, ply + 1, move, False)
                score = -score
            else:
                r = 0
                if depth >= 2 and is_quiet:
                    r  = _LMR[min(depth, 63) * 64 + min(move_count, 63)]
                    r -= int(pv_node); r -= int(improving); r += int(cut_node)
                    r -= history[move.from_square * 64 + move.to_square] // 8_000
                    if winning_side_to_move:
                        r -= 1
                    r  = max(0, min(r, new_depth - 1))
                score, _ = self._search(board, new_depth - r, -alpha - 1, -alpha,
                                        ply + 1, move, True)
                score = -score
                if r > 0 and score > alpha:
                    score, _ = self._search(board, new_depth, -alpha - 1, -alpha,
                                            ply + 1, move, not cut_node)
                    score = -score
                if alpha < score < beta:
                    score, _ = self._search(board, new_depth, -beta, -alpha,
                                            ply + 1, move, False)
                    score = -score

            board.pop()

            if score > best_val: best_val = score; best_move = move
            if score > alpha: alpha = score
            if alpha >= beta:
                if is_quiet:
                    self._update_quiet_stats(board, move, depth, ply, prev_move, quiets_searched)
                break
            if is_quiet: quiets_searched.append(move)

        if move_count == 0:
            if excluded_move: return alpha, None
            if in_check:
                return -MATE_SCORE + ply, None
            draw_eval = static_eval if eval_valid else self.evaluate(board, key)
            return self._draw_score_from_eval(draw_eval), None

        if not excluded_move:
            flag = UPPER if best_val <= orig_alpha else LOWER if best_val >= beta else EXACT
            if len(self.tt) > TT_SIZE: self.tt.clear()
            self.tt[key] = (depth, best_val, flag, best_move)

        return best_val, best_move

    # Iterative deepening with aspiration windows
    def get_best_move(self, board: chess.Board) -> chess.Move | None:
        legal = list(board.legal_moves)
        if not legal: return None
        if len(legal) == 1: return legal[0]

        self._cancel_event.clear()
        self._stop_search = False
        self.nodes = 0
        self._age_history()

        if self.time_limit_ms is not None:
            self._deadline_ns = time.perf_counter_ns() + int(self.time_limit_ms * 1_000_000)
            max_depth = MAX_PLY - 1
        else:
            self._deadline_ns = None
            max_depth = self.depth

        best_move = None
        best_score = 0

        for d in range(1, max_depth + 1):
            if self._stop_search: break
            if self._time_is_up():
                self._stop_search = True
                break

            if d <= 5 or best_move is None:
                score, move = self._search(board, d, -MATE_SCORE, MATE_SCORE, 0)
                if not self._stop_search and move is not None:
                    best_move = move; best_score = score
                continue

            # Aspiration windows
            delta = 40
            lo, hi = best_score - delta, best_score + delta
            fail_high_move  = None
            fail_high_score = -MATE_SCORE

            while True:
                score, move = self._search(board, d, lo, hi, 0)
                if self._stop_search:
                    if fail_high_move is not None and fail_high_score > best_score:
                        best_move = fail_high_move; best_score = fail_high_score
                    break
                if score <= lo:
                    lo = max(-MATE_SCORE, lo - delta); delta *= 2
                elif score >= hi:
                    if move is not None: fail_high_move = move; fail_high_score = score
                    hi = min(MATE_SCORE, hi + delta); delta *= 2
                else:
                    if move is not None:
                        best_move = move; best_score = score
                    elif fail_high_move is not None:
                        best_move = fail_high_move; best_score = fail_high_score
                    break

        self._deadline_ns = None
        return best_move or legal[0]
    