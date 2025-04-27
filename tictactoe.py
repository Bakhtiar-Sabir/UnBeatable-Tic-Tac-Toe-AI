import time

class TicTacToe:
    """Class representing the Tic-Tac-Toe game board and rules"""
    
    def __init__(self):
        """Initialize empty 3x3 board and track winner"""
        self.board = [' ' for _ in range(9)]  # 1D list representing 3x3 grid
        self.current_winner = None  # Tracks if there's a winner

    def print_board(self):
        """Display the current board state"""
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        """Show position numbers for player reference"""
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        """Return list of available move indices (empty squares)"""
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        """Check if there are empty squares remaining"""
        return ' ' in self.board

    def num_empty_squares(self):
        """Count number of empty squares"""
        return self.board.count(' ')

    def make_move(self, square, letter):
        """Attempt to make a move, return True if valid"""
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        """Check if the last move resulted in a win"""
        # Check row
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all(spot == letter for spot in row):
            return True
        
        # Check column
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all(spot == letter for spot in column):
            return True
        
        # Check diagonals (only if square is even number - corner or center)
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]  # Top-left to bottom-right
            if all(spot == letter for spot in diagonal1):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]  # Top-right to bottom-left
            if all(spot == letter for spot in diagonal2):
                return True
        return False

def minimax(position, maximizing_player, player_letter, memo=None):
    """Minimax algorithm implementation with memoization for optimization"""
    if memo is None:
        memo = {}
    
    board_key = tuple(position.board)
    if board_key in memo:
        return memo[board_key]
    
    opponent_letter = 'O' if player_letter == 'X' else 'X'
    
    # Base cases
    if position.current_winner == player_letter:
        return {'position': None, 'score': 1 * (position.num_empty_squares() + 1)}
    elif position.current_winner == opponent_letter:
        return {'position': None, 'score': -1 * (position.num_empty_squares() + 1)}
    elif not position.empty_squares():
        return {'position': None, 'score': 0}

    if maximizing_player:
        best = {'position': None, 'score': -float('inf')}
        for possible_move in position.available_moves():
            position.make_move(possible_move, player_letter)
            sim_score = minimax(position, False, player_letter, memo)
            
            # Undo move
            position.board[possible_move] = ' '
            position.current_winner = None
            
            sim_score['position'] = possible_move
            if sim_score['score'] > best['score']:
                best = sim_score
        memo[board_key] = best
        return best
    else:
        best = {'position': None, 'score': float('inf')}
        for possible_move in position.available_moves():
            position.make_move(possible_move, opponent_letter)
            sim_score = minimax(position, True, player_letter, memo)
            
            # Undo move
            position.board[possible_move] = ' '
            position.current_winner = None
            
            sim_score['position'] = possible_move
            if sim_score['score'] < best['score']:
                best = sim_score
        memo[board_key] = best
        return best

def alphabeta(position, maximizing_player, player_letter, alpha=-float('inf'), beta=float('inf'), memo=None):
    """Alpha-Beta pruning optimized minimax with memoization"""
    if memo is None:
        memo = {}
    
    board_key = tuple(position.board)
    if board_key in memo:
        return memo[board_key]
    
    opponent_letter = 'O' if player_letter == 'X' else 'X'
    
    # Base cases
    if position.current_winner == player_letter:
        return {'position': None, 'score': 1 * (position.num_empty_squares() + 1)}
    elif position.current_winner == opponent_letter:
        return {'position': None, 'score': -1 * (position.num_empty_squares() + 1)}
    elif not position.empty_squares():
        return {'position': None, 'score': 0}

    if maximizing_player:
        best = {'position': None, 'score': -float('inf')}
        for possible_move in position.available_moves():
            position.make_move(possible_move, player_letter)
            sim_score = alphabeta(position, False, player_letter, alpha, beta, memo)
            
            # Undo move
            position.board[possible_move] = ' '
            position.current_winner = None
            
            sim_score['position'] = possible_move
            if sim_score['score'] > best['score']:
                best = sim_score
            
            alpha = max(alpha, best['score'])
            if beta <= alpha:
                break
        memo[board_key] = best
        return best
    else:
        best = {'position': None, 'score': float('inf')}
        for possible_move in position.available_moves():
            position.make_move(possible_move, opponent_letter)
            sim_score = alphabeta(position, True, player_letter, alpha, beta, memo)
            
            # Undo move
            position.board[possible_move] = ' '
            position.current_winner = None
            
            sim_score['position'] = possible_move
            if sim_score['score'] < best['score']:
                best = sim_score
            
            beta = min(beta, best['score'])
            if beta <= alpha:
                break
        memo[board_key] = best
        return best

def compare_algorithms():
    """Compare performance of minimax vs alpha-beta pruning"""
    game = TicTacToe()
    
    # Warm-up runs to account for Python's startup overhead
    minimax(game, True, 'X')
    alphabeta(game, True, 'X')
    
    # Test Minimax
    start = time.perf_counter()
    for _ in range(10):
        minimax(game, True, 'X')
    minimax_time = time.perf_counter() - start
    
    # Test Alpha-Beta
    start = time.perf_counter()
    for _ in range(10):
        alphabeta(game, True, 'X')
    alphabeta_time = time.perf_counter() - start
    
    print(f"\nPerformance Comparison:")
    print(f"- Minimax average time: {minimax_time/10:.6f}s")
    print(f"- Alpha-Beta average time: {alphabeta_time/10:.6f}s")
    print(f"- Improvement: {((minimax_time-alphabeta_time)/minimax_time)*100:.2f}% faster")

def play(game, x_player, o_player, print_game=True):
    """Main game loop"""
    if print_game:
        game.print_board_nums()
    
    letter = 'X'  # Starting player
    while game.empty_squares():
        if letter == 'O':
            square = o_player(game, letter)
        else:
            square = x_player(game, letter)
        
        if game.make_move(square, letter):
            if print_game:
                print(f"{letter} makes a move to square {square}")
                game.print_board()
                print('')
            
            if game.current_winner:
                if print_game:
                    print(f"{letter} wins!")
                return letter
            
            letter = 'O' if letter == 'X' else 'X'
    
    if print_game:
        print("It's a tie!")
    return None

def human_player(game, letter):
    """Handle human player input"""
    while True:
        square = input(f"{letter}'s turn. Input move (0-8): ")
        try:
            val = int(square)
            if val not in game.available_moves():
                raise ValueError
            return val
        except ValueError:
            print("Invalid square. Try again.")

def minimax_player(game, letter):
    """AI player using minimax algorithm"""
    time.sleep(0.5)  # Small delay for better UX
    return minimax(game, True, letter)['position']

def alphabeta_player(game, letter):
    """AI player using alpha-beta pruning"""
    time.sleep(0.5)  # Small delay for better UX
    return alphabeta(game, True, letter)['position']

def main():
    """Main menu and game setup"""
    print("\nTIC-TAC-TOE AI")
    print("1. Play Against Minimax AI")
    print("2. Play Against Alpha-Beta AI")
    print("3. Compare Algorithm Performance")
    print("4. Watch AI vs AI Match")
    print("5. Exit")
    
    while True:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '5':
            print("Thanks for playing!")
            break
            
        game = TicTacToe()
        
        if choice == '1':
            print("\nHuman (X) vs Minimax AI (O)")
            play(game, human_player, minimax_player)
        elif choice == '2':
            print("\nHuman (X) vs Alpha-Beta AI (O)")
            play(game, human_player, alphabeta_player)
        elif choice == '3':
            compare_algorithms()
        elif choice == '4':
            print("\nMinimax AI (X) vs Alpha-Beta AI (O)")
            play(game, minimax_player, alphabeta_player)
        else:
            print("Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()