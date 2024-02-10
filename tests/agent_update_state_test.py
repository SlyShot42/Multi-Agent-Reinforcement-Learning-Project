from resources import MainGame, Agent, EndPnt, GameObj
from resources.config import CELL_NUMBER, CELL_SIZE
from pygame.math import Vector2


game = MainGame(2)
game1 = MainGame(1)


def update_states(game):
    for bot in game.bots:
        bot.update_state(game)


def move_all_to_positions(game):
    for i in range(game.num_agents):
        game.move_to_positions(
            game.bots[i].front, game.bots[i].back, game.end_pts[i].pos
        )


def move_all_to_extractions(game):
    for i in range(game.num_agents):
        game.move_to_extractions(
            game.bots[i].front, game.bots[i].back, game.end_pts[i].pos
        )


def show_vectors(game):
    GameObj.display_vectors(*game.bots, *game.end_pts)


def test_update_state_down():
    # tests danger straight, danger right, end point left,
    # end point straight, end point behind
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(5, 4))
    game.bots[1].place(Vector2(5, 6), Vector2(4, 6))
    game.end_pts[0].place(Vector2(10, 10))
    game.end_pts[1].place(Vector2(10, 11))
    move_all_to_extractions(game)
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
    assert game.bots[1].state == [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]

    # tests danger left, no danger, end point right
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(5, 4))
    game.bots[1].place(Vector2(6, 6), Vector2(6, 5))
    game.end_pts[0].place(Vector2(2, 10))
    game.end_pts[1].place(Vector2(2, 11))
    move_all_to_extractions(game)
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0]
    assert game.bots[1].state == [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]


def test_update_state_up():
    # tests danger straight, danger left, end point right,
    # end point straight, end point behind
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(5, 6))
    game.bots[1].place(Vector2(5, 4), Vector2(6, 4))
    game.end_pts[0].place(Vector2(10, 1))
    game.end_pts[1].place(Vector2(10, 2))
    move_all_to_extractions(game)
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
    assert game.bots[1].state == [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1]

    # tests danger right, no danger, end point left
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(5, 6))
    game.bots[1].place(Vector2(4, 6), Vector2(4, 7))
    game.end_pts[0].place(Vector2(1, 1))
    game.end_pts[1].place(Vector2(2, 2))
    move_all_to_extractions(game)
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    assert game.bots[1].state == [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]


def test_update_state_left():
    # tests danger straight, danger right, end point left,
    # end point straight, end point right
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(6, 5))
    game.bots[1].place(Vector2(4, 5), Vector2(4, 6))
    game.end_pts[0].place(Vector2(1, 1))
    game.end_pts[1].place(Vector2(2, 2))
    move_all_to_extractions(game)
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]
    assert game.bots[1].state == [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

    # tests danger left, no danger, end point behind
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(6, 5))
    game.bots[1].place(Vector2(6, 4), Vector2(7, 4))
    game.end_pts[0].place(Vector2(10, 10))
    game.end_pts[1].place(Vector2(11, 11))
    move_all_to_extractions(game)
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    assert game.bots[1].state == [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]


def test_update_state_right():
    # tests danger straight, danger left, end point left,
    # end point straight, end point behind
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(4, 5))
    game.bots[1].place(Vector2(6, 5), Vector2(6, 6))
    game.end_pts[0].place(Vector2(1, 1))
    game.end_pts[1].place(Vector2(2, 2))
    move_all_to_extractions(game)
    update_states(game)
    print("\ntest_update_state_down\n")
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert game.bots[1].state == [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]

    # tests danger right, no danger, end point right
    move_all_to_positions(game)
    game.bots[0].place(Vector2(5, 5), Vector2(4, 5))
    game.bots[1].place(Vector2(4, 4), Vector2(3, 4))
    game.end_pts[0].place(Vector2(2, 10))
    game.end_pts[1].place(Vector2(2, 11))
    move_all_to_extractions(game)
    update_states(game)
    print(f"bot 1 state: {game.bots[0].state}")
    print(f"bot 2 state: {game.bots[1].state}")
    # show_vectors(game)
    assert game.bots[0].state == [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]
    assert game.bots[1].state == [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1]


def test_update_state_straight_boundary():
    move_all_to_positions(game1)
    game1.bots[0].place(
        Vector2(CELL_NUMBER - 1, int(CELL_NUMBER / 2)),
        Vector2(CELL_NUMBER - 2, int(CELL_NUMBER / 2)),
    )
    game1.end_pts[0].place(Vector2(0, int(CELL_NUMBER / 2)))
    move_all_to_extractions(game1)
    update_states(game1)
    print("\ntest_update_state_straight_boundary\n")
    print(f"bot state: {game1.bots[0].state}")
    # show_vectors(game1)
    assert game1.bots[0].state == [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]


def test_update_state_left_boundary():
    move_all_to_positions(game1)
    game1.bots[0].place(
        Vector2(int(CELL_NUMBER / 2), 0), Vector2(int(CELL_NUMBER / 2) - 1, 0)
    )
    game1.end_pts[0].place(Vector2(CELL_NUMBER - 1, CELL_NUMBER - 1))
    move_all_to_extractions(game1)
    GameObj.display_vectors(*game1.bots, *game1.end_pts)
    update_states(game1)
    print("\ntest_update_state_left_boundary\n")
    print(f"bot state: {game1.bots[0].state}")
    # show_vectors(game1)
    assert game1.bots[0].state == [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0]


def test_update_state_right_boundary():
    move_all_to_positions(game1)
    game1.bots[0].place(
        Vector2(int(CELL_NUMBER / 2), CELL_NUMBER - 1),
        Vector2(int(CELL_NUMBER / 2) - 1, CELL_NUMBER - 1),
    )
    game1.end_pts[0].place(Vector2(CELL_NUMBER - 1, 0))
    move_all_to_extractions(game1)
    update_states(game1)
    print("\ntest_update_state_right_boundary\n")
    print(f"bot state: {game1.bots[0].state}")
    # show_vectors(game1)
    assert game1.bots[0].state == [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
