from engine.simpleAlphaBetaEngine import DefaultMinMaxEngine
from engine.nn_engine import NeuralNetEngine
from engine.stockfish_engine import StockfishEngine
from config import *

def get_engine(engine_name: str, minMaxDepth=AI_MINMAX_DEPTH, stockFishElo=STOCKFISH_ELO, stockfishTimeLimit=STOCKFISH_TIME_LIMIT):
    engine_name = engine_name.lower()
    
    if engine_name == 'minimax':
        return DefaultMinMaxEngine(minMaxDepth)
    
    elif engine_name == 'neural' or engine_name == 'nn':
        return NeuralNetEngine(minMaxDepth)
    
    elif engine_name == 'stockfish':
        return StockfishEngine(stockFishElo, stockfishTimeLimit)
    
    else:
        raise ValueError(f"Unknown engine: {engine_name}. Choose 'minimax', 'neural', or 'stockfish'")
