import csv
import os
import time
import chess
import chess.pgn

import engine.nn_engine as nn_engine_module
from engine.nn_engine import NeuralNetEngine
from engine.stockfish_engine import StockfishEngine


AI_CONFIGS = [
    {"name": "ai5fen96c16b", "model_path": r"saved_ai\ai5\96c16b\ai5fen96c16b.pt", "depth": None},
    {"name": "ai5lichess96c16bAllGames", "model_path": r"saved_ai\ai5\96c16b\ai5lichess96c16bAllGames.pt", "depth": None},
    {"name": "ai5lichess96c16b13milPositions", "model_path": r"saved_ai\ai5\96c16b\ai5lichess96c16b13milPositions.pt", "depth": None},
    {"name": "ai5combined96c16b", "model_path": r"saved_ai\ai5\96c16b\ai5combined96c16b.pt", "depth": None},
]
STOCKFISH_SETTINGS = [
    # {"elo": 2600, "time_limit": 1},
    {"elo": 2500, "time_limit": 1},
    # {"elo": 2400, "time_limit": 1},
    # {"elo": 2300, "time_limit": 1},
]
NEURAL_NET_TIME_LIMIT = 2.5
TOTAL_GAMES = 100
OUTPUT_DIR_PREFIX = "logs/ai5_2_5sTime_96c16b/"

SUMMARY_TXT = "match_summaries.txt"
MAX_PLIES = 250
RESULT_DRAW = "1/2-1/2"
CSV_HEADERS = ["game", "ai_color", "result", "ai_score", "plies", "duration_sec", "avg_ai_move_time", "termination", "moves_pgn"]
OUTPUT_DIR = None


def get_ai_score(result: str, is_white: bool) -> float:
    if result == "1-0":
        return 1.0 if is_white else 0.0
    if result == "0-1":
        return 0.0 if is_white else 1.0
    return 0.5

def get_csv_path(ai_name: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{ai_name}.csv")

def get_summary_path() -> str:
    return os.path.join(OUTPUT_DIR, SUMMARY_TXT)

def init_output_files():
    global OUTPUT_DIR
    OUTPUT_DIR = f"{OUTPUT_DIR_PREFIX}_{stockfish_cfg['elo']}elo_{stockfish_cfg['time_limit']}s"
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cfg in AI_CONFIGS:
        csv_path = get_csv_path(cfg["name"])
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    summary_path = get_summary_path()
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("AI vs Stockfish match summaries\n")
            f.write(f"Total games per AI: {TOTAL_GAMES}\n")
            f.write(f"Stockfish settings: {STOCKFISH_SETTINGS}\n")
            f.write("-" * 60 + "\n")


def load_existing_match_results(ai_name: str) -> list[dict]:
    csv_path = get_csv_path(ai_name)
    if not os.path.exists(csv_path):
        return []

    loaded_results = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_idx = int(row["game"])
            if game_idx < 1 or game_idx > TOTAL_GAMES:
                continue
            loaded_results.append({
                "game": game_idx,
                "ai_color": row["ai_color"],
                "result": row["result"],
                "ai_score": float(row["ai_score"]),
                "plies": int(row["plies"]),
                "duration": float(row["duration_sec"]),
                "avg_ai_move_time": float(row["avg_ai_move_time"]),
                "termination": row["termination"],
                "moves": row["moves_pgn"],
            })

    return loaded_results


def append_game_result(ai_name: str, data: dict):
    with open(get_csv_path(ai_name), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data["game"],
            data["ai_color"],
            data["result"],
            f"{data['ai_score']:.1f}",
            data["plies"],
            f"{data['duration']:.3f}",
            f"{data['avg_ai_move_time']:.4f}",
            data["termination"],
            data["moves"]
        ])
    print(
        f"[{ai_name}] {data['game']}/{TOTAL_GAMES} | {data['ai_color']:<5} | {data['result']} "
        f"| {data['plies']} plies | {data['duration']:.2f}s | avg ai move {data['avg_ai_move_time']:.3f}s"
    )

def moves_to_pgn_text(board: chess.Board) -> str:
    game = chess.pgn.Game()
    game.add_line(board.move_stack)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    return game.accept(exporter).strip()

def build_match_summary_lines(ai_name: str, ai_depth: int, games: list[dict]) -> list[str]:
    total_score = sum(game["ai_score"] for game in games)
    wins = sum(1 for game in games if game["ai_score"] == 1.0)
    draws = sum(1 for game in games if game["ai_score"] == 0.5)
    losses = sum(1 for game in games if game["ai_score"] == 0.0)

    avg_plies = sum(game["plies"] for game in games) / TOTAL_GAMES
    avg_duration = sum(game["duration"] for game in games) / TOTAL_GAMES
    avg_ai_move_time = (sum(game["avg_ai_move_time"] for game in games) / TOTAL_GAMES)

    white_games = [game for game in games if game["ai_color"] == "white"]
    black_games = [game for game in games if game["ai_color"] == "black"]
    white_score = sum(game["ai_score"] for game in white_games)
    black_score = sum(game["ai_score"] for game in black_games)

    return [
        f"[{ai_name}] Match summary",
        f"Depth: {ai_depth}",
        f"AI time limit: {NEURAL_NET_TIME_LIMIT:.2f}s/move",
        f"Games: {TOTAL_GAMES}",
        f"Score: {total_score:.1f}/{TOTAL_GAMES} ({(100.0 * total_score / TOTAL_GAMES):.1f}%)",
        f"W-D-L: {wins}-{draws}-{losses}",
        f"Score as White: {white_score:.1f}/{len(white_games)}",
        f"Score as Black: {black_score:.1f}/{len(black_games)}",
        f"Avg plies: {avg_plies:.1f}",
        f"Avg game duration: {avg_duration:.2f}s",
        f"Avg model move time: {avg_ai_move_time:.3f}s",
    ]

def print_match_summary(ai_name: str, ai_depth: int, games: list[dict]):
    lines = build_match_summary_lines(ai_name, ai_depth, games)
    print()
    for line in lines:
        print(line)

def append_match_summary_to_file(ai_name: str, ai_depth: int, games: list[dict]):
    lines = build_match_summary_lines(ai_name, ai_depth, games)
    with open(get_summary_path(), "a") as f:
        f.write("\n")
        f.write("\n".join(lines))
        f.write("\n")
        f.write("-" * 60 + "\n")


def play_single_game(game_idx: int, ai_engine, stockfish_engine : StockfishEngine, ai_is_white: bool):
    board = chess.Board()
    start_time = time.perf_counter()
    ai_think_total = 0.0
    ai_move_count = 0

    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
        is_ai_turn = (board.turn == chess.WHITE) == ai_is_white
        if is_ai_turn:
            t0 = time.perf_counter()
            move = ai_engine.get_best_move(board)
            ai_think_total += time.perf_counter() - t0
            ai_move_count += 1
        else:
            move = stockfish_engine.get_best_move(board)
        if move is None:
            break
        board.push(move)

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        result = board.result(claim_draw=True)
        termination = outcome.termination.name
    else:
        result = RESULT_DRAW
        termination = "MAX_PLIES"

    plies = len(board.move_stack)
    duration = time.perf_counter() - start_time
    avg_ai_move_time = (ai_think_total / ai_move_count) if ai_move_count > 0 else 0.0

    return {
        "game": game_idx,
        "ai_color": "white" if ai_is_white else "black",
        "result": result,
        "ai_score": get_ai_score(result, ai_is_white),
        "plies": plies,
        "duration": duration,
        "avg_ai_move_time": avg_ai_move_time,
        "termination": termination,
        "moves": moves_to_pgn_text(board),
    }

def run_match(ai_config: dict, stockfish_engine: StockfishEngine):
    ai_name = ai_config["name"]
    print(f"{ai_name} started")

    match_results = load_existing_match_results(ai_name)
    if match_results:
        print(f"[{ai_name}] Resuming from game {len(match_results) + 1}/{TOTAL_GAMES}.")
    if len(match_results) >= TOTAL_GAMES:
        print(f"[{ai_name}] Already complete. Skipping.")
        return

    nn_engine_module.AI_MODEL_PATH = ai_config["model_path"]
    ai_engine = NeuralNetEngine(ai_config["depth"], time_limit_ms= NEURAL_NET_TIME_LIMIT * 1000)

    for game_idx in range(len(match_results) + 1, TOTAL_GAMES + 1):
        ai_is_white = game_idx <= TOTAL_GAMES // 2
        result = play_single_game(game_idx, ai_engine, stockfish_engine, ai_is_white)
        match_results.append(result)
        append_game_result(ai_name, result)

    print_match_summary(ai_name, ai_config["depth"], match_results)
    append_match_summary_to_file(ai_name, ai_config["depth"], match_results)


if __name__ == "__main__":
    for stockfish_cfg in STOCKFISH_SETTINGS:
        stockfish_engine = StockfishEngine(**stockfish_cfg)
        if not stockfish_engine.available:
            raise RuntimeError(stockfish_engine.unavailable_reason)

        init_output_files()

        try:
            for config in AI_CONFIGS:
                run_match(config, stockfish_engine)
        finally:
            stockfish_engine.close()
