from PySide6.QtCore import QThread, Signal

class MinimaxThread(QThread):

    searchComplete = Signal(int)

    def __init__(self, chessboard, parent = None):
        super().__init__(parent=parent)
        self.chessboard = chessboard

    def run(self):
        chessboard = self.chessboard.inner.clone()
        action = chessboard.minimax_best_move()
        self.searchComplete.emit(action)