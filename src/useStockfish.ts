import { useEffect, useRef, useState, useCallback } from 'react';

export interface StockfishEval {
  score: number;       // centipawns from white's perspective (e.g. +150 = white up 1.5 pawns)
  mate: number | null; // moves to mate (positive = white mates, negative = black mates)
  depth: number;
  bestMove: string;
  thinking: boolean;
}

const INITIAL: StockfishEval = { score: 0, mate: null, depth: 0, bestMove: '', thinking: false };

export function useStockfish(fen: string, targetDepth = 16): StockfishEval {
  const [evalState, setEvalState] = useState<StockfishEval>(INITIAL);
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef(false);
  // Track whose turn it is so we can flip the score to always be from White's POV
  const sideRef = useRef<'w' | 'b'>('w');
  // Count of pending `stop` commands whose bestmove responses should be ignored
  const pendingStopsRef = useRef(0);
  // Whether a search is currently running (so we know whether to send stop)
  const searchingRef = useRef(false);

  // Initialize the engine once
  useEffect(() => {
    const worker = new Worker('/stockfish.js');

    worker.onmessage = (e: MessageEvent) => {
      const line = typeof e.data === 'string' ? e.data : '';

      if (line === 'uciok') {
        worker.postMessage('isready');
      }
      if (line === 'readyok') {
        readyRef.current = true;
      }

      // Parse "info depth X score cp Y" or "info depth X score mate Y"
      if (line.startsWith('info depth')) {
        const depthMatch = line.match(/depth (\d+)/);
        const cpMatch = line.match(/score cp (-?\d+)/);
        const mateMatch = line.match(/score mate (-?\d+)/);

        if (depthMatch) {
          const depth = parseInt(depthMatch[1]);
          // Stockfish reports score relative to side-to-move; flip if black
          const flip = sideRef.current === 'b' ? -1 : 1;
          if (cpMatch) {
            setEvalState(prev => ({
              ...prev,
              score: parseInt(cpMatch[1]) * flip,
              mate: null,
              depth,
              thinking: true,
            }));
          } else if (mateMatch) {
            const mateVal = parseInt(mateMatch[1]) * flip;
            setEvalState(prev => ({
              ...prev,
              mate: mateVal,
              score: mateVal > 0 ? 10000 : -10000,
              depth,
              thinking: true,
            }));
          }
        }
      }

      // Parse "bestmove e2e4"
      if (line.startsWith('bestmove')) {
        if (pendingStopsRef.current > 0) {
          // This bestmove is from a stop command, not a completed search — ignore it
          pendingStopsRef.current -= 1;
        } else {
          const move = line.split(' ')[1] || '';
          searchingRef.current = false;
          setEvalState(prev => ({ ...prev, bestMove: move, thinking: false }));
        }
      }
    };

    worker.postMessage('uci');
    workerRef.current = worker;

    return () => {
      worker.postMessage('quit');
      worker.terminate();
    };
  }, []);

  // Evaluate whenever FEN changes
  const pendingFen = useRef<string | null>(null);

  const evaluate = useCallback((fenStr: string) => {
    const worker = workerRef.current;
    if (!worker || !readyRef.current) {
      // Engine not ready yet — store for later
      pendingFen.current = fenStr;
      return;
    }

    pendingFen.current = null;
    sideRef.current = fenStr.split(' ')[1] === 'b' ? 'b' : 'w';

    if (searchingRef.current) {
      // Cancel the active search; track that we'll receive a bestmove to ignore
      pendingStopsRef.current += 1;
      worker.postMessage('stop');
    }

    worker.postMessage(`position fen ${fenStr}`);
    worker.postMessage(`go depth ${targetDepth}`);
    searchingRef.current = true;
    setEvalState(prev => ({ ...prev, thinking: true }));
  }, [targetDepth]);

  useEffect(() => {
    evaluate(fen);
    // If engine wasn't ready, poll until it is
    if (!readyRef.current) {
      const interval = setInterval(() => {
        if (readyRef.current && pendingFen.current) {
          evaluate(pendingFen.current);
          clearInterval(interval);
        }
      }, 200);
      return () => clearInterval(interval);
    }
  }, [fen, evaluate]);

  return evalState;
}
