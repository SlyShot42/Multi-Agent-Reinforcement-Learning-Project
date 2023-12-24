from main import MAIN_GAME, Agent, END_PNT, GAME_OBJ, Action
from pygame.math import Vector2

game = None

def setup_function():
    global game
    game = MAIN_GAME(2)

def test_update_state_down():
    # tests danger straight, danger right, end point left, 
    # end point straight, end point behind
    game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
    game.bots[1] = Agent(Vector2(5,6),Vector2(4,6))
    game.end_pts[0] = END_PNT(Vector2(10,10))
    game.end_pts[1] = END_PNT(Vector2(10,11))
    game.update_state()
    print('\ntest_update_state_down\n')
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [1,0,0,0,0,0,1,1,0,1,0]
    assert game.bots[1].state == [0,1,0,0,1,0,0,0,1,1,0]

    # tests danger left, no danger, end point right
    game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
    game.bots[1] = Agent(Vector2(6,6),Vector2(6,5))
    game.end_pts[0] = END_PNT(Vector2(2,10))
    game.end_pts[1] = END_PNT(Vector2(2,11))
    game.update_state()
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [0,1,0,0,0,0,1,0,1,1,0]
    assert game.bots[1].state == [0,0,0,0,0,0,1,0,1,1,0]

def test_update_state_up():
    # tests danger straight, danger left, end point right, 
    # end point straight, end point behind
    game.bots[0] = Agent(Vector2(5,5),Vector2(5,6))
    game.bots[1] = Agent(Vector2(5,4),Vector2(6,4))
    game.end_pts[0] = END_PNT(Vector2(10,1))
    game.end_pts[1] = END_PNT(Vector2(10,2))
    game.update_state()
    print('\ntest_update_state_down\n')
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [1,0,0,0,0,1,0,0,1,1,0]
    assert game.bots[1].state == [0,1,0,1,0,0,0,0,1,0,1]

    # tests danger right, no danger, end point left
    game.bots[0] = Agent(Vector2(5,5),Vector2(5,6))
    game.bots[1] = Agent(Vector2(4,6),Vector2(4,7))
    game.end_pts[0] = END_PNT(Vector2(1,1))
    game.end_pts[1] = END_PNT(Vector2(2,2))
    game.update_state()
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [0,0,0,0,0,1,0,1,0,1,0]
    assert game.bots[1].state == [0,0,1,0,0,1,0,1,0,1,0]

def test_update_state_left():
    # tests danger straight, danger right, end point left, 
    # end point straight, end point right
    game.bots[0] = Agent(Vector2(5,5),Vector2(6,5))
    game.bots[1] = Agent(Vector2(4,5),Vector2(4,6))
    game.end_pts[0] = END_PNT(Vector2(1,1))
    game.end_pts[1] = END_PNT(Vector2(2,2))
    game.update_state()
    print('\ntest_update_state_down\n')
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [1,0,0,1,0,0,0,0,1,1,0]
    assert game.bots[1].state == [0,0,1,0,0,1,0,1,0,1,0]

    # tests danger left, no danger, end point behind
    game.bots[0] = Agent(Vector2(5,5),Vector2(6,5))
    game.bots[1] = Agent(Vector2(6,4),Vector2(7,4))
    game.end_pts[0] = END_PNT(Vector2(10,10))
    game.end_pts[1] = END_PNT(Vector2(11,11))
    game.update_state()
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [0,0,0,1,0,0,0,1,0,0,1]
    assert game.bots[1].state == [0,1,0,1,0,0,0,1,0,0,1]

def test_update_state_right():
    # tests danger straight, danger left, end point left, 
    # end point straight, end point behind
    game.bots[0] = Agent(Vector2(5,5),Vector2(4,5))
    game.bots[1] = Agent(Vector2(6,5),Vector2(6,6))
    game.end_pts[0] = END_PNT(Vector2(1,1))
    game.end_pts[1] = END_PNT(Vector2(2,2))
    game.update_state()
    print('\ntest_update_state_down\n')
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [1,0,0,0,1,0,0,1,0,0,1]
    assert game.bots[1].state == [0,1,0,0,0,1,0,1,0,1,0]

    # tests danger right, no danger, end point right
    game.bots[0] = Agent(Vector2(5,5),Vector2(4,5))
    game.bots[1] = Agent(Vector2(4,4),Vector2(3,4))
    game.end_pts[0] = END_PNT(Vector2(2,10))
    game.end_pts[1] = END_PNT(Vector2(2,11))
    game.update_state()
    print(f'bot 1 state: {game.bots[0].state}')
    print(f'bot 2 state: {game.bots[1].state}')
    assert game.bots[0].state == [0,0,0,0,1,0,0,0,1,0,1]
    assert game.bots[1].state == [0,0,1,0,1,0,0,0,1,0,1]

