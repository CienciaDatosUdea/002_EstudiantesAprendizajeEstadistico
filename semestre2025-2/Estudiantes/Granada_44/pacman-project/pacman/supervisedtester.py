import sys
import os

#Fix import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#Now import from the correct locations
try:
    from game import Directions, GameStateData, Game
    from layout import getLayout
    import textDisplay
    import graphicsDisplay
except ImportError:
    import __main__
    from pacman.game import Directions, GameStateData, Game
    from pacman.layout import getLayout
    from pacman import textDisplay
    from pacman import graphicsDisplay

class SupervisedTester:
    def __init__(self):
        self.results = defaultdict(list)
        self.layouts = [
            'testClassic',
            'smallClassic', 
            'mediumClassic',
            'capsuleClassic',
            'minimaxClassic'
        ]
    
    def run_single_test(self, layout_name, num_games=10, display=None):
        from game import runGames
        
        #Set up the game
        layout = getLayout(layout_name)
        if layout is None:
            print(f"Layout {layout_name} not found")
            return None
        
        #Create agent
        agent = SupervisedAgent()
        
        #Use null display if none specified
        if display is None:
            display = textDisplay.NullGraphics()
        
        print(f"\n🧪 Testing Supervised Agent on {layout_name} ({num_games} games)")
        print("-" * 50)
        
        #Run games
        games = runGames(
            layout=layout,
            pacman=agent,
            ghosts=['RandomGhost', 'RandomGhost', 'RandomGhost', 'RandomGhost'],
            display=display,
            numGames=num_games,
            record=False,
            catchExceptions=True
        )
        
        # Collect results
        scores = []
        wins = 0
        for i, game in enumerate(games):
            score = game.state.getScore()
            scores.append(score)
            is_win = game.state.isWin()
            wins += 1 if is_win else 0
            
            # Print result for this game
            result = "WIN" if is_win else "LOSS"
            print(f"Game {i+1}: Score = {score:6.1f} ({result})")
        
        #Calculate statistics
        avg_score = sum(scores) / len(scores)
        win_rate = (wins / num_games) * 100
        
        #Store results
        self.results[layout_name] = {
            'scores': scores,
            'avg_score': avg_score,
            'win_rate': win_rate,
            'wins': wins,
            'total_games': num_games,
            'min_score': min(scores),
            'max_score': max(scores)
        }
        
        return self.results[layout_name]
    
    def run_all_tests(self, num_games_per_layout=10, display=None):
        
        for layout in self.layouts:
            self.run_single_test(layout, num_games_per_layout, display)
        
        #Print overall summary
        self.print_summary()
    
    def export_results(self, filename="supervised_results.csv"):
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layout', 'Game Number', 'Score', 'Win', 'Average Score', 'Win Rate'])
            
            for layout, result in self.results.items():
                for i, score in enumerate(result['scores']):
                    is_win = 1 if i < result['wins'] else 0
                    writer.writerow([
                        layout,
                        i + 1,
                        score,
                        is_win,
                        result['avg_score'],
                        result['win_rate']
                    ])
        
        print(f"\nResults exported to {filename}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Supervised Learning Pacman Agent')
    parser.add_argument('-l', '--layout', type=str, default='testClassic',
                       help='Layout to test on')
    parser.add_argument('-n', '--numGames', type=int, default=10,
                       help='Number of games to run')
    parser.add_argument('--all', action='store_true',
                       help='Test on all layouts')
    parser.add_argument('--visual', action='store_true',
                       help='Show visual display (slower)')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')
    
    args = parser.parse_args()
    
    tester = SupervisedTester()
    
    #Choose display
    if args.visual:
        display = graphicsDisplay.PacmanGraphics()
    else:
        display = textDisplay.NullGraphics()
    
    # Run tests
    if args.all:
        tester.run_all_tests(num_games_per_layout=args.numGames, display=display)
    else:
        tester.run_single_test(args.layout, args.numGames, display)
    
    #Export if requested
    if args.export:
        tester.export_results(args.export)
    else:
        tester.export_results()

if __name__ == '__main__':
    main()