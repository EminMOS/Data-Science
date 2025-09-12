def print_board(board):
    """Prints the tic-tac-toe board (positions 1..9)."""
    def cell(i):
        return board[i] if board[i] != " " else str(i)
    print()
    print(f" {cell(1)} | {cell(2)} | {cell(3)} ")
    print("---+---+---")
    print(f" {cell(4)} | {cell(5)} | {cell(6)} ")
    print("---+---+---")
    print(f" {cell(7)} | {cell(8)} | {cell(9)} ")
    print()

def check_win(board, symbol):
    """Return True if `symbol` has a winning line on `board`."""
    winning_lines = [
        (1,2,3), (4,5,6), (7,8,9),
        (1,4,7), (2,5,8), (3,6,9),
        (1,5,9), (3,5,7)
    ]
    for a, b, c in winning_lines:
        if board[a] == symbol and board[b] == symbol and board[c] == symbol:
            return True
    return False

def main():
    nameX = input('Enter name for Player X: ').strip() or "Player X"
    nameO = input('Enter name for Player O: ').strip() or "Player O"

    # board indices 1..9, index 0 unused
    board = [" "] * 10
    currentPlayer = 'X'
    currentName = nameX
    moves = 0

    print("\nStarting Tic-Tac-Toe!")
    print_board(board)

    while True:
        prompt = f"{currentName} ({currentPlayer}), choose a position (1-9): "
        pos_input = input(prompt).strip()

        # Validate numeric input
        try:
            pos = int(pos_input)
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 9.")
            continue

        if pos < 1 or pos > 9:
            print("Invalid position, try again.")
            continue

        if board[pos] != " ":
            print("Position already taken, choose another.")
            continue

        board[pos] = currentPlayer
        moves += 1

        print_board(board)

        if check_win(board, currentPlayer):
            print(f"{currentName} ({currentPlayer}) wins!")
            break

        if moves == 9:
            print("It's a draw!")
            break

        # Switch player
        if currentPlayer == 'X':
            currentPlayer = 'O'
            currentName = nameO
        else:
            currentPlayer = 'X'
            currentName = nameX

if __name__ == "__main__":
    main()