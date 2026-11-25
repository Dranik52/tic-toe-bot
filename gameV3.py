import json
import os
import random
import sys

import pygame


WIDTH, HEIGHT = 540, 740
BOARD_SIZE = 3
CELL_SIZE = WIDTH // BOARD_SIZE
BOARD_HEIGHT = CELL_SIZE * BOARD_SIZE
FPS = 60
Q_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")
BG_COLOR = (245, 245, 245)
LINE_COLOR = (50, 50, 50)
X_COLOR = (210, 60, 60)
O_COLOR = (40, 110, 200)
TEXT_COLOR = (30, 30, 30)
BUTTON_COLOR = (225, 232, 242)
BUTTON_HOVER = (200, 216, 237)
WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
        (2, 5, 8), (0, 4, 8), (2, 4, 6))

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Крестики-нолики — Q-Learning")
clock = pygame.time.Clock()
symbol_font = pygame.font.SysFont("arial", 96, bold=True)
title_font = pygame.font.SysFont("arial", 34, bold=True)
status_font = pygame.font.SysFont("arial", 23)
small_font = pygame.font.SysFont("arial", 18)


def new_game():
    return [None] * 9, "X", None


def check_win(board):
    for a, b, c in WINS:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return "Draw" if all(cell is not None for cell in board) else None


def moves(board):
    return [i for i, cell in enumerate(board) if cell is None]


def state_key(board, player):
    return "".join("0" if cell is None else "1" if cell == player else "2" for cell in board)


def empty_tables():
    return {"X": {}, "O": {}}


def load_q():
    try:
        with open(Q_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "X" in data or "O" in data:
            return {p: {s: {int(a): float(v) for a, v in acts.items()}
                        for s, acts in data.get(p, {}).items()} for p in ("X", "O")}
        shared = {s: {int(a): float(v) for a, v in acts.items()} for s, acts in data.items()}
        return {p: {s: dict(acts) for s, acts in shared.items()} for p in ("X", "O")}
    except (OSError, ValueError, TypeError):
        return empty_tables()


def save_q(q):
    temp = Q_FILE + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(q, f, separators=(",", ":"))
    os.replace(temp, Q_FILE)


def value(q, state, action):
    return q.get(state, {}).get(action, 0.0)


def choose_move(board, player, q, epsilon=0.0):
    legal = moves(board)
    if random.random() < epsilon:
        return random.choice(legal)
    state = state_key(board, player)
    vals = [value(q[player], state, a) for a in legal]
    best = max(vals)
    return random.choice([a for a, v in zip(legal, vals) if v == best])


def train_episode(q, epsilon):
    board, player, result = new_game()
    last = {}
    while result is None:
        state = state_key(board, player)
        action = choose_move(board, player, q, epsilon)
        last[player] = (state, action)
        board[action] = player
        result = check_win(board)
        if result == player:
            reward, future = 1.0, 0.0
        elif result == "Draw":
            reward, future = 0.0, 0.0
        else:
            other = "O" if player == "X" else "X"
            next_state = state_key(board, other)
            future = max(value(q[other], next_state, a) for a in moves(board))
            reward = 0.0
        actions = q[player].setdefault(state, {})
        old = actions.get(action, 0.0)
        actions[action] = old + 0.18 * (reward - future - old)
        if result == "Draw":
            for p, (s, a) in last.items():
                old_last = q[p].setdefault(s, {}).get(a, 0.0)
                q[p][s][a] = old_last + 0.18 * (0.0 - old_last)
        elif result is not None:
            loser = "O" if result == "X" else "X"
            s, a = last[loser]
            old_last = q[loser].setdefault(s, {}).get(a, 0.0)
            q[loser][s][a] = old_last + 0.18 * (-1.0 - old_last)
        player = "O" if player == "X" else "X"
    return board, result


def draw_text(text, font, color, center):
    surf = font.render(text, True, color)
    screen.blit(surf, surf.get_rect(center=center))


def draw_button(rect, text, hover=False):
    pygame.draw.rect(screen, BUTTON_HOVER if hover else BUTTON_COLOR, rect, border_radius=10)
    pygame.draw.rect(screen, LINE_COLOR, rect, 2, border_radius=10)
    draw_text(text, status_font, TEXT_COLOR, rect.center)


def draw_board(board):
    screen.fill(BG_COLOR)
    for i in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, BOARD_HEIGHT), 4)
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE), 4)
    pygame.draw.line(screen, LINE_COLOR, (0, BOARD_HEIGHT), (WIDTH, BOARD_HEIGHT), 3)
    pad = 28
    for idx, mark in enumerate(board):
        if mark is None:
            continue
        row, col = divmod(idx, 3)
        x, y = col * CELL_SIZE, row * CELL_SIZE
        if mark == "X":
            pygame.draw.line(screen, X_COLOR, (x + pad, y + pad),
                             (x + CELL_SIZE - pad, y + CELL_SIZE - pad), 10)
            pygame.draw.line(screen, X_COLOR, (x + CELL_SIZE - pad, y + pad),
                             (x + pad, y + CELL_SIZE - pad), 10)
        else:
            pygame.draw.circle(screen, O_COLOR, (x + CELL_SIZE // 2, y + CELL_SIZE // 2),
                               CELL_SIZE // 2 - pad, 10)


def menu(q, notice=""):
    buttons = [(pygame.Rect(100, 215, 340, 62), "Обучение: бот против бота", "train"),
               (pygame.Rect(100, 295, 340, 62), "Играть против бота", "play"),
               (pygame.Rect(100, 375, 340, 62), "Сбросить обучение", "reset")]
    while True:
        screen.fill(BG_COLOR)
        draw_text("Крестики-нолики", title_font, TEXT_COLOR, (WIDTH // 2, 105))
        draw_text("Q-Learning", status_font, TEXT_COLOR, (WIDTH // 2, 150))
        count = sum(len(q.get(p, {})) for p in ("X", "O"))
        draw_text(f"Сохранённых состояний X и O: {count}", small_font,
                  TEXT_COLOR, (WIDTH // 2, 185))
        if notice:
            draw_text(notice, small_font, TEXT_COLOR, (WIDTH // 2, 465))
        for rect, label, _ in buttons:
            draw_button(rect, label, rect.collidepoint(pygame.mouse.get_pos()))
        draw_text("Выберите режим", small_font, TEXT_COLOR, (WIDTH // 2, 535))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q(q); pygame.quit(); sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, _, action in buttons:
                    if rect.collidepoint(event.pos):
                        if action == "train":
                            return setup_training()
                        if action == "play":
                            if not any(q.get(p) for p in ("X", "O")):
                                notice = "Сначала запустите обучение"
                            else:
                                return setup_play()
                        if action == "reset":
                            q.clear(); q.update(empty_tables()); save_q(q)
                            notice = "Обучение сброшено"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()
        clock.tick(FPS)


def setup_training():
    options = [(100_000, "100 000 партий"), (1_000_000, "1 000 000 партий"),
               (5_000_000, "5 000 000 партий")]
    while True:
        screen.fill(BG_COLOR)
        draw_text("Обучение двух ботов", title_font, TEXT_COLOR, (WIDTH // 2, 130))
        draw_text("Выберите число партий", small_font, TEXT_COLOR, (WIDTH // 2, 180))
        rects = []
        for i, (amount, label) in enumerate(options):
            rect = pygame.Rect(120, 230 + i * 70, 300, 55); rects.append((rect, amount))
            draw_button(rect, label, rect.collidepoint(pygame.mouse.get_pos()))
        draw_text("Esc — назад", small_font, TEXT_COLOR, (WIDTH // 2, 480))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, amount in rects:
                    if rect.collidepoint(event.pos): return amount
        clock.tick(FPS)


def train_screen(q, total):
    done, latest_board, latest_result = 0, [None] * 9, None
    while done < total:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q(q); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                save_q(q); return
        batch = min(1200, total - done)
        for i in range(batch):
            eps = max(0.04, 0.55 * (1 - (done + i) / total))
            latest_board, latest_result = train_episode(q, eps)
        done += batch
        draw_board(latest_board)
        draw_text("Боты обучаются", title_font, TEXT_COLOR, (WIDTH // 2, BOARD_HEIGHT + 38))
        draw_text(f"Партии: {done:,} / {total:,}".replace(",", " "), status_font,
                  TEXT_COLOR, (WIDTH // 2, BOARD_HEIGHT + 78))
        bar = pygame.Rect(60, BOARD_HEIGHT + 105, WIDTH - 120, 20)
        pygame.draw.rect(screen, (215, 215, 215), bar, border_radius=8)
        fill = bar.copy(); fill.width = int(bar.width * done / total)
        pygame.draw.rect(screen, (60, 150, 90), fill, border_radius=8)
        result = "Ничья" if latest_result == "Draw" else f"Последняя партия: выиграл {latest_result}"
        draw_text(result, small_font, TEXT_COLOR, (WIDTH // 2, BOARD_HEIGHT + 145))
        draw_text("Esc — сохранить и вернуться в меню", small_font, TEXT_COLOR,
                  (WIDTH // 2, BOARD_HEIGHT + 175))
        pygame.display.flip(); clock.tick(FPS)
    save_q(q)


def setup_play():
    options = [(pygame.Rect(85, 225, 370, 55), "Играть за X (первый ход)", "X"),
               (pygame.Rect(85, 295, 370, 55), "Играть за O (второй ход)", "O")]
    while True:
        screen.fill(BG_COLOR)
        draw_text("Выберите сторону", title_font, TEXT_COLOR, (WIDTH // 2, 125))
        for rect, label, _ in options:
            draw_button(rect, label, rect.collidepoint(pygame.mouse.get_pos()))
        draw_text("Esc — назад", small_font, TEXT_COLOR, (WIDTH // 2, 410))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, _, side in options:
                    if rect.collidepoint(event.pos): return side, setup_difficulty()
        clock.tick(FPS)


def setup_difficulty():
    options = [(0.22, "Лёгкая"), (0.08, "Средняя"), (0.0, "Сложная")]
    while True:
        screen.fill(BG_COLOR)
        draw_text("Сложность бота", title_font, TEXT_COLOR, (WIDTH // 2, 120))
        rects = []
        for i, (eps, label) in enumerate(options):
            rect = pygame.Rect(115, 205 + i * 70, 310, 55); rects.append((rect, eps))
            draw_button(rect, label, rect.collidepoint(pygame.mouse.get_pos()))
        draw_text("На лёгкой и средней сложности бывают случайные ходы.", small_font,
                  TEXT_COLOR, (WIDTH // 2, 430))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return 0.0
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, eps in rects:
                    if rect.collidepoint(event.pos): return eps
        clock.tick(FPS)


def play(q, human, epsilon):
    board, player, result = new_game()
    bot = "O" if human == "X" else "X"
    bot_at = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_q(q); pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return
                if event.key == pygame.K_r: board, player, result = new_game(); bot_at = 0
            if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and result is None
                    and player == human):
                x, y = event.pos
                idx = (y // CELL_SIZE) * 3 + x // CELL_SIZE if y < BOARD_HEIGHT and x < WIDTH else None
                if idx is not None and board[idx] is None:
                    board[idx] = human; result = check_win(board)
                    if result is None: player = bot; bot_at = pygame.time.get_ticks() + 250
        if result is None and player == bot and pygame.time.get_ticks() >= bot_at:
            board[choose_move(board, bot, q, epsilon)] = bot
            result = check_win(board)
            if result is None: player = human
        draw_board(board)
        if result is None: msg = "Бот думает..." if player == bot else f"Ваш ход ({human})"
        elif result == "Draw": msg = "Ничья! Нажмите R для новой игры"
        else: msg = "Вы выиграли! R — ещё раз" if result == human else "Бот выиграл! R — ещё раз"
        draw_text(msg, status_font, TEXT_COLOR, (WIDTH // 2, BOARD_HEIGHT + 38))
        draw_text("Esc — меню", small_font, TEXT_COLOR, (WIDTH // 2, BOARD_HEIGHT + 77))
        pygame.display.flip(); clock.tick(FPS)


def main():
    q = load_q()
    notice = ""
    while True:
        mode = menu(q, notice); notice = ""
        if mode is None: continue
        if isinstance(mode, int):
            train_screen(q, mode); notice = "Обучение завершено и сохранено"
        elif mode is not None:
            human, epsilon = mode
            if epsilon is not None: play(q, human, epsilon)


if __name__ == "__main__":
    main()
