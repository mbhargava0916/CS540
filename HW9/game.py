import random
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def run_challenge_test(self):
        """ Set to True if you would like to run gradescope against the challenge AI!
        Leave as False if you would like to run the gradescope tests faster for debugging.
        You can still get full credit with this set to False
        """ 
        return False

    def make_move(self, state):
        """ Selects a (row, col) space for the next move using the minimax algorithm.
        Args:
            state (list of lists): the current state of the game as saved in this TeekoPlayer object.
        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase).
        """
        # Detect drop phase
        piece_count = sum(row.count('b') + row.count('r') for row in state)
        drop_phase = piece_count < 8

        # Use minimax to find the best move
        depth = 3
        best_score = float('-inf')
        best_move = None

        for successor in self.succ(state, self.my_piece):
            score = self.min_value(successor, depth - 1)
            if score > best_score:
                best_score = score
                best_move = successor

        # Convert the best move into the required format
        move = []
        if drop_phase:
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ' and best_move[row][col] == self.my_piece:
                        move.append((row, col))
                        return move
        else:
            # Movement phase logic
            for row in range(5):
                for col in range(5):
                    if state[row][col] == self.my_piece and best_move[row][col] == ' ':
                        source = (row, col)
                    elif state[row][col] == ' ' and best_move[row][col] == self.my_piece:
                        dest = (row, col)
            move.append(dest)
            move.append(source)
            return move

    def succ(self, state, piece):
        """ Generate all legal successor states.
        Args:
            state (list of lists): the current state of the game.
            piece (str): the piece ('b' or 'r') to generate successors for.
        Returns:
            list: a list of successor states.
        """
        successors = []
        piece_count = sum(row.count('b') + row.count('r') for row in state)

        if piece_count < 8:  # Drop phase
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        new_state = [row[:] for row in state]
                        new_state[row][col] = piece
                        successors.append(new_state)
        else:  # Movement phase
            for row in range(5):
                for col in range(5):
                    if state[row][col] == piece:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                new_row, new_col = row + dr, col + dc
                                if 0 <= new_row < 5 and 0 <= new_col < 5 and state[new_row][new_col] == ' ':
                                    new_state = [row[:] for row in state]
                                    new_state[row][col] = ' '
                                    new_state[new_row][new_col] = piece
                                    successors.append(new_state)
        return successors

    def heuristic_game_value(self, state):
        """ Evaluate non-terminal states using a heuristic.
        Args:
            state (list of lists): the current state of the game.
        Returns:
            float: a heuristic score between -1 and 1.
        """
        terminal_value = self.game_value(state)
        if terminal_value != 0:  # Terminal state
            return terminal_value

        # Evaluate based on potential wins
        my_score = 0
        opp_score = 0

        # Count consecutive pieces in rows, columns, diagonals, and 2x2 boxes
        for row in state:
            my_score += max(row.count(self.my_piece) for row in state)
            opp_score += max(row.count(self.opp) for row in state)

        for col in range(5):
            column = [state[row][col] for row in range(5)]
            my_score += max(column.count(self.my_piece) for col in range(5))
            opp_score += max(column.count(self.opp) for col in range(5))

        # Normalize scores between -1 and 1
        total_score = (my_score - opp_score) / 16.0
        return total_score

    def game_value(self, state):
        """ Checks the current board status for a win condition.
        Args:
            state (list of lists): the current state of the game.
        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner.
        """
        # Check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i] == self.my_piece else -1

        # Check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # Check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # Check / diagonal wins
        for row in range(2):
            for col in range(3, 5):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # Check 2x2 box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row][col+1] == state[row+1][col+1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # No winner yet

    def max_value(self, state, depth):
        """ Maximizing function for the minimax algorithm.
        Args:
            state (list of lists): the current state of the game.
            depth (int): the remaining depth to explore.
        Returns:
            float: the maximum score achievable from this state.
        """
        if depth == 0 or self.game_value(state) != 0:
            return self.heuristic_game_value(state)

        alpha = float('-inf')
        for successor in self.succ(state, self.my_piece):
            alpha = max(alpha, self.min_value(successor, depth - 1))
        return alpha

    def min_value(self, state, depth):
        """ Minimizing function for the minimax algorithm.
        Args:
            state (list of lists): the current state of the game.
            depth (int): the remaining depth to explore.
        Returns:
            float: the minimum score achievable from this state.
        """
        if depth == 0 or self.game_value(state) != 0:
            return self.heuristic_game_value(state)

        beta = float('inf')
        for successor in self.succ(state, self.opp):
            beta = min(beta, self.max_value(successor, depth - 1))
        return beta

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
        """
        # Validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # Make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece.
        Args:
            move (list): a list of move tuples.
            piece (str): the piece ('b' or 'r') to place on the board.
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board.
        """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # Drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:
        # Get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # Update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # Move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:
        # Get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # Update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()