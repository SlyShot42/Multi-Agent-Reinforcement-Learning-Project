import unittest
from main import MAIN_GAME, Agent, END_PNT, GAME_OBJ, Action
from pygame.math import Vector2

class TestMainGame(unittest.TestCase):

    def setUp(self) -> None:
        self.game = MAIN_GAME(2)

    def test_update_state_no_collision(self):
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(5,6),Vector2(5,7))
        board = 0
        for i, bot in enumerate(self.game.bots): 
            board += bot.vector
        print('\ntest_update_state_no_collision\n')
        print(board)
        self.game.update_state()
        print(f'\ncollision matrix: {self.game.collision}')
        self.assertNotIn(1,self.game.collision)

    def test_update_state_head_to_head_collision(self):
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(5,5),Vector2(5,6))
        board = 0
        for i, bot in enumerate(self.game.bots): 
            board += bot.vector
        print('\ntest_update_state_head_to_head_collision\n')
        print(board)
        self.game.update_state()
        print(f'\ncollision matrix: {self.game.collision}')
        self.assertIn(1,self.game.collision)

    def test_update_state_head_to_tail_collision(self):
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(5,4),Vector2(6,4))
        board = 0
        for i, bot in enumerate(self.game.bots): 
            board += bot.vector
        print('\ntest_update_state_head_to_tail_collision\n')
        print(board)
        self.game.update_state()
        print(f'\ncollision matrix: {self.game.collision}')
        self.assertIn(1,self.game.collision)

    def test_update_state_zero_terminates(self):
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.end_pts[0] = END_PNT(Vector2(10,6))
        self.game.bots[1] = Agent(Vector2(6,5),Vector2(6,4))
        self.game.end_pts[1] = END_PNT(Vector2(11,8))
        self.game.update_state()
        print('\ntest_update_state_zero_terminates\n')
        print(f'\ndones array: {self.game.dones}')
        self.assertNotIn(1,self.game.dones)

    def test_update_state_single_terminates(self):
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.end_pts[0] = END_PNT(Vector2(5,5))
        self.game.bots[1] = Agent(Vector2(6,5),Vector2(6,4))
        self.game.end_pts[1] = END_PNT(Vector2(11,8))
        self.game.update_state()
        print('\ntest_update_state_single_terminates\n')
        print(f'\ndones array: {self.game.dones}')
        self.assertIn(1,self.game.dones)

    def test_update_state_multi_terminates(self):
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.end_pts[0] = END_PNT(Vector2(5,5))
        self.game.bots[1] = Agent(Vector2(6,5),Vector2(6,4))
        self.game.end_pts[1] = END_PNT(Vector2(6,5))
        print('\ntest_update_state_multi_terminates\n')
        self.game.update_state()
        print(f'\ndones array: {self.game.dones}')
        self.assertIn(1,self.game.dones)

    '''
    TODO: first thing in the morning, write tests for the following:
    '''
    def test_update_state_bot_state_danger_straight(self):
        '''
        TODO: this test will use the game update_state method
        to check if the danger straight state is being updated correctly
        '''
        print('\ntest_update_state_bot_state_danger_straight\n')

        '''
        TODO: HEAD TO HEAD
        '''
        self.game.bots[0] = Agent(Vector2(5,5),Vector2(5,4))
        self.game.bots[1] = Agent(Vector2(5,6),Vector2(5,7))
        self.game.end_pts[0] = END_PNT(Vector2(10,10))
        self.game.end_pts[1] = END_PNT(Vector2(10,11))
        GAME_OBJ.display_vectors(self.game.bots)
        self.game.update_state()
        self.assertEqual(self.game.bots[0].state,[1,0,0,0,0,0,1,])
        '''
        TODO: HEAD TO TAIL
        '''

    
    def test_update_state_bot_state_danger_left(self):
        pass

    def test_update_state_bot_state_danger_right(self):
        pass

    def test_update_state_bot_state_direction_left(self):
        pass

    def test_update_state_bot_state_direction_right(self):
        pass

    def test_update_state_bot_state_direction_up(self):
        pass

    def test_update_state_bot_state_direction_down(self):
        pass



if __name__ == '__main__':
    unittest.main()
        