import { useEffect, useRef, useState, useCallback } from 'react';

export interface StockfishEval {
  score: number;       // centipawns from white's perspective (e.g. +150 = white up 1.5 pawns)
  mate: number | null; // moves to mate (positive = white mates, negative = black mates)
  depth: number;
  bestMove: string;
  thinking: boolean;
}

const INITIAL: StockfishEval = { score: 0, mate: null, depth: 0, bestMove: '', thinking: false };

/**
 * Manages Stockfish evaluation using the lichess protocol pattern:
 * - Never send `go` while a previous search is active
 * - `bestmove` is the only signal that clears the active search
 * - Rapid position changes collapse into a single queued search
 */
export function useStockfish(fen: string, targetDepth = 16): StockfishEval {
  const [evalState, setEvalState] = useState<StockfishEval>(INITIAL);
  const workerRef = useRef<Worker | null>(null);
  const readyRef = useRef(false);

  // Current active search (null = engine is idle)
  const workRef = useRef<{ fen: string; stopRequested: boolean } | null>(null);
  // Queued search to start after the current one finishes
  const nextWorkRef = useRef<string | null>(null);

  const swapWork = useCallback(() => {
    const worker = workerRef.current;
    if (!worker || !readyRef.current) return;
    if (workRef.current) return; // wait for current search to finish

    const nextFen = nextWorkRef.current;
    nextWorkRef.current = null;

    if (nextFen) {
      const side = nextFen.split(' ')[1] === 'b' ? 'b' : 'w';
      workRef.current = { fen: nextFen, stopRequested: false };
      // Store side on the work object for score flipping
      (workRef.current as any).side = side;
      worker.postMessage(`position fen ${nextFen}`);
      worker.postMessage(`go depth ${targetDepth}`);
      setEvalState(prev => ({ ...prev, thinking: true }));
    }
  }, [targetDepth]);

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
        swapWork();
      }

      // Parse "info depth X score cp Y" or "info depth X score mate Y"
      // Suppress if the current search was asked to stop
      if (line.startsWith('info depth') && workRef.current && !workRef.current.stopRequested) {
        const depthMatch = line.match(/depth (\d+)/);
        const cpMatch = line.match(/score cp (-?\d+)/);
        const mateMatch = line.match(/score mate (-?\d+)/);

        if (depthMatch) {
          const depth = parseInt(depthMatch[1]);
          const flip = (workRef.current as any).side === 'b' ? -1 : 1;
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

      // bestmove = search finished. Clear work and start the next queued search.
      if (line.startsWith('bestmove')) {
        const wasActive = workRef.current && !workRef.current.stopRequested;
        workRef.current = null;
        if (wasActive) {
          const move = line.split(' ')[1] || '';
          setEvalState(prev => ({ ...prev, bestMove: move, thinking: false }));
        }
        swapWork();
      }
    };

    worker.postMessage('uci');
    workerRef.current = worker;

    return () => {
      worker.postMessage('quit');
      worker.terminate();
    };
  }, [swapWork]);

  // Queue a new evaluation: stop current search if needed, swap when ready
  const compute = useCallback((fenStr: string) => {
    nextWorkRef.current = fenStr;

    if (workRef.current && !workRef.current.stopRequested) {
      workRef.current.stopRequested = true;
      workerRef.current?.postMessage('stop');
    }

    swapWork();
  }, [swapWork]);

  useEffect(() => {
    compute(fen);
  }, [fen, compute]);

  return evalState;
}
