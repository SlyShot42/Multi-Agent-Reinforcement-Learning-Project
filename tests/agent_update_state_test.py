from resources import MainGame, Agent, EndPnt
from resources.config import CELL_NUMBER, CELL_SIZE
from pygame.math import Vector2


game = MainGame(2)
game1 = MainGame(1)


def update_states(obj):
    for bot in obj.bots:
        bot.update_state(game)


def test_update_state_down():
    # tests danger straight, danger right, end point left,
    # end point straight, end point behind
    game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 4))
    game.bots[1] = Agent(Vector2(5, 6), Vector2(4, 6))
    game.end_pts[0] = EndPnt(Vector2(10, 10))
    game.end_pts[1] = EndPnt(Vector2(10, 11))
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
    assert game.bots[1].state == [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]

    # tests danger left, no danger, end point right
    game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 4))
    game.bots[1] = Agent(Vector2(6, 6), Vector2(6, 5))
    game.end_pts[0] = EndPnt(Vector2(2, 10))
    game.end_pts[1] = EndPnt(Vector2(2, 11))
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    assert game.bots[1].state == [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]


def test_update_state_up():
    # tests danger straight, danger left, end point right,
    # end point straight, end point behind
    game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 6))
    game.bots[1] = Agent(Vector2(5, 4), Vector2(6, 4))
    game.end_pts[0] = EndPnt(Vector2(10, 1))
    game.end_pts[1] = EndPnt(Vector2(10, 2))
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
    assert game.bots[1].state == [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]

    # tests danger right, no danger, end point left
    game.bots[0] = Agent(Vector2(5, 5), Vector2(5, 6))
    game.bots[1] = Agent(Vector2(4, 6), Vector2(4, 7))
    game.end_pts[0] = EndPnt(Vector2(1, 1))
    game.end_pts[1] = EndPnt(Vector2(2, 2))
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    assert game.bots[1].state == [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def test_update_state_left():
    # tests danger straight, danger right, end point left,
    # end point straight, end point right
    game.bots[0] = Agent(Vector2(5, 5), Vector2(6, 5))
    game.bots[1] = Agent(Vector2(4, 5), Vector2(4, 6))
    game.end_pts[0] = EndPnt(Vector2(1, 1))
    game.end_pts[1] = EndPnt(Vector2(2, 2))
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    assert game.bots[1].state == [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

    # tests danger left, no danger, end point behind
    game.bots[0] = Agent(Vector2(5, 5), Vector2(6, 5))
    game.bots[1] = Agent(Vector2(6, 4), Vector2(7, 4))
    game.end_pts[0] = EndPnt(Vector2(10, 10))
    game.end_pts[1] = EndPnt(Vector2(11, 11))
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    assert game.bots[1].state == [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]


def test_update_state_right():
    # tests danger straight, danger left, end point left,
    # end point straight, end point behind
    game.bots[0] = Agent(Vector2(5, 5), Vector2(4, 5))
    game.bots[1] = Agent(Vector2(6, 5), Vector2(6, 6))
    game.end_pts[0] = EndPnt(Vector2(1, 1))
    game.end_pts[1] = EndPnt(Vector2(2, 2))
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert game.bots[1].state == [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

    # tests danger right, no danger, end point right
    game.bots[0] = Agent(Vector2(5, 5), Vector2(4, 5))
    game.bots[1] = Agent(Vector2(4, 4), Vector2(3, 4))
    game.end_pts[0] = EndPnt(Vector2(2, 10))
    game.end_pts[1] = EndPnt(Vector2(2, 11))
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    assert game.bots[0].state == [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    assert game.bots[1].state == [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]


def test_update_state_straight_boundary():
    game1.bots[0] = Agent(
        Vector2(CELL_NUMBER - 1, int(CELL_NUMBER / 2)),
        Vector2(CELL_NUMBER - 2, int(CELL_NUMBER / 2)),
    )
    game1.end_pts[0] = EndPnt(Vector2(0, int(CELL_NUMBER / 2)))
    update_states(game1)
    print("\ntest_update_state_straight_boundary\n")
    print(f"bot state: {game1.bots[0].state}")
    assert game.bots[0].state == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]


def test_update_state_left_boundary():
    game1.bots[0] = Agent(
        Vector2(int(CELL_NUMBER / 2), 0), Vector2(int(CELL_NUMBER / 2) - 1, 0)
    )
    game1.end_pts[0] = EndPnt(Vector2(CELL_NUMBER - 1, CELL_NUMBER - 1))
    update_states(game1)
    print("\ntest_update_state_left_boundary\n")
    print(f"bot state: {game1.bots[0].state}")
    assert game.bots[0].state == [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]


def test_update_state_right_boundary():
    game1.bots[0] = Agent(
        Vector2(int(CELL_NUMBER / 2), CELL_NUMBER - 1),
        Vector2(int(CELL_NUMBER / 2) - 1, CELL_NUMBER - 1),
    )
    game1.end_pts[0] = EndPnt(Vector2(CELL_NUMBER - 1, 0))
    update_states(game1)
    print("\ntest_update_state_right_boundary\n")
    print(f"bot state: {game1.bots[0].state}")
    assert game.bots[0].state == [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
