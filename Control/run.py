# Imports
import os
import time
from colorama import Back, Fore, Style



""" To run the cmd"""
# os.system("cmd")

""" To store the output"""
# output = os.popen(f"cat {__file__}").read()
# print(output)

#--------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------


# Triggers in the environment
waiting_triggers = {
        "Running": "The job is currently running",
        "ContainerCreating": "The Kubernetes cluster is creating the docker container. It might take some time, especially when a new docker image is pulled from the docker hub site.",
        "Pending": "The scheduler is waiting for a node which has the requested resources available."
    }
terminating_triggers = {
    "Error":"The job terminated with an error",
    "Completed":"The job has successfully completed",
    "ImagePullBackOff": "It occurs when their is an issue in pulling the specified docker image either due to internet connectivity or wrong image name.",
    "ErrImagePull":"Error while pulling the image, try checking your network connectivity",
}

class KubernetesRunner():
    def __init__(self, root_path):
        self.ROOT_PATH = root_path
        self.start_script = os.path.join(root_path, "start.sh")
        self.status_script = os.path.join(root_path, "status.sh")
        self.log_script = os.path.join(root_path, "log.sh")

        
    def forward(self):

        os.system(f"bash {self.start_script}") # starting the container
        print(Fore.CYAN + f"Starting execution...")
        print(f"Status\tTime")
        start_time = time.time()
        status_counter = 1
        while(True):
            # sl = 
            status = os.popen(f"bash {self.status_script}").read().split()[7] # position of status values in the output string
            
            current_time = time.time()
            if status in waiting_triggers:
                print(Fore.YELLOW, status_counter, status, f":\t{round(current_time - start_time, 2)}s")
                time.sleep(1)
                status_counter += 1
            else:
                color_code = None
                if status == "Error":
                    color_code = Style.BRIGHT + Back.RED + Fore.YELLOW
                elif status == "Completed":
                    color_code = Style.BRIGHT + Back.GREEN + Fore.BLUE
                else:
                    color_code = Style.NORMAL + Back.BLUE + Fore.YELLOW
                
                print(color_code, status_counter, status, f":\t{terminating_triggers[status]}")
                execution_time = round(current_time - start_time, 2)
                print(Style.RESET_ALL + f"Execution time: {execution_time}s")
                print(f"")
                print(Fore.WHITE + f"Log file contents:")
                print(f"--------------------" + Style.RESET_ALL)
                os.system(f"bash {self.log_script}")
                print(f"")
                return execution_time

class InputCode():
    def __init__(self, code="blank"):
        self.code = "blank" 
    
    def forward(self, input):
        # Rules for the input
        return self.blank_code(input)

    def blank_code(self, input):
        return len(input) == 1


def main():
    import sys

    ic = InputCode() # Caller function to input code class
    if not ic.forward(sys.argv):
        print(f"valid options: ")
        readme = open("Readme.txt", "r")
        readme = readme.readlines()
        for line in readme:
            print(line)
        exit()
    
    kr = KubernetesRunner(root_path = "Lib/")
    exec_time = kr.forward()

if __name__ == "__main__":
    main()

        


