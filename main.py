import os
import sys
import getopt
from cartpole import CartPoleLearner

c = CartPoleLearner()

def run_random():
   c.random_test()

def run_new_policy():
   c.train()
   c.test()

def run_from_policy(path:str):
   c.from_policy_test(path)

def main(argv):
   keys, _ = getopt.getopt(argv,"hp:r",["help","policy=","random"])

   if not keys:
      run_new_policy()

   for key, value in keys:
      if key in ("-h", "--help"):
         print("-----------------------------------------------------------------------------")
         print ('To run a new policy: python main.py')
         print ('To run from a policy file: python main.py -p <csv file from policies folder>')
         print ('To run randomly: python main.py -r')
         print("-----------------------------------------------------------------------------")

      elif key in ("-p", "--policy"):
         if not os.path.isfile(f'policies/{value}'):
            print ("Not a valid file")
            return
         
         run_from_policy(f'policies/{value}')
         return
      
      elif key in ("-r", "--random"):
         run_random()
         return

if __name__ == "__main__":
   main(sys.argv[1:])