"""
Description: The main code for running.
Editor: Jinbiao Zhu
Date: 19-10-2024
"""
import datetime
import json
import pprint
import re
import time
from typing import List, Union

from langchain.agents import AgentOutputParser
from langchain.agents import LLMSingleActionAgent
from langchain.agents.agent import AgentFinish, AgentExecutor, AgentAction
from langchain.callbacks import get_openai_callback
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.prompts.chat import HumanMessage
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import Tool

from global_parameters import GlobalParam
from m1509 import M1509v2
from sim import *

os.environ["OPENAI_API_KEY"] = GlobalParam.OPENAI_API_KEY_for_Agent

running_info = dict()
robot = M1509v2()


def generate_scenario_description(name: str):
    return "There is a table in front of " + name + \
        " with an orange, a banana, a computer mouse, a clock, a bleach, a water bottle, and a food can on it."


class AnalyzeHumanIntention:
    def __init__(self):
        self.llm = OpenAI(model_name=GlobalParam.OpenAI_chat_model,
                          temperature=GlobalParam.Default_temperature,
                          max_tokens=GlobalParam.Default_max_tokens)
        self.text = """
Please play the role of an excellent human-query analyzer. I am Alice, a human, and I will tell you my query.
Please analyze my query and select the most suitable single or multiple objects based on the actual scenario.
The actual scenario you are in: {scenario_description}
Please answer in the format of string.
When you don't know how to answer, please directly output an empty string.
Your answer, apart from a string, should not contain any other content!
Here are two examples:
----------
Query: I need supplies for my weekend DIY project on building a computer.
Answer: Give Alice a computer mouse.
----------
Let's start!
Query: {query}
Answer:
"""
        self.template = PromptTemplate(template=self.text, input_variables=["query", "scenario_description"])

    def run(self, user_input: str) -> str:
        full_prompt = self.template.format(query=user_input,
                                           scenario_description=generate_scenario_description("you"))
        print(full_prompt)
        response = self.llm(full_prompt)
        running_info["query"] = user_input
        running_info["intentions"] = response
        return response


class GenerateCodeChain:
    def __init__(self):
        self.llm = OpenAI(model_name=GlobalParam.OpenAI_chat_model,
                          temperature=GlobalParam.Default_temperature,
                          max_tokens=GlobalParam.Default_max_tokens)
        self.prompt_template_text = """
You will play the role of an excellent Python programmer.
You will generate the Python-based robot control codes for a 6 degree-of-freedom DOOSAN M1509 robotic arm with a gripper
based on the task provided.
You must use the following pieces of abstract class to complete the code. 
----------------------------------------------
class M1509:
    def __init__(self):
        # Function:
        #     This method realizes the initialization of the robot and the perceptual system.
        # Note:
        #     This method is typically used at the beginning of a code script to instantiate a class.
        # For example:
        #     robot = M1509()
        #     ...

    def move_horizontally(self, object_name):
        # Function:
        #     This method realizes the horizontal movement of the end effector.
        #     It moves the end effector on the horizontal plane to realize the horizontal movement of the end effector
        #         to the same "x" and "y" coordinates of "object_name" based on the world reference frame.
        # Input:
        #     object_name: Name of the object.
        # Note:
        #     The position of the end effector is changed, the posture remains unchanged,
        #         and the holding state of the gripper remains unchanged.
        # For example:
        #     ...
        #     robot.observe()
        #     robot.move_horizontally("object")
        #     ...

    def move_vertically(self, object_name):
        # Function:
        #     This method realizes the vertical movement of the end effector of the robot.
        #     It moves the end effector on the vertical axis to realize the vertical movement of the end effector to
        #         the same 'z' coordinate of "object_name" based on the world reference frame.
        # Input:
        #     object_name: Name of the object.
        # Note:
        #     The position of the end effector is changed, the posture remains unchanged, and the holding state of the
        #     gripper remains unchanged.
        # For example:
        #     ...
        #     robot.observe()
        #     robot.move_vertically("object")
        #     ...

    def grasp(self, onoff: bool):
        # Function:
        #     This method realizes the opening and closing of the gripper of the robot arm according to the "on_off",
        #         and then realizes the robot arm to grasp or release the object.
        # Input:
        #     on_off: True if a robot want to grasp something, and False if a robot want to release something.
        # Note:
        #     The position of the end effector remains unchanged, the posture remains unchanged,
        #         and the holding state of the gripper is changed.
        # For example:
        #     ...
        #     robot.observe()
        #     robot.move_vertically("object")
        #     robot.grasp(True)
        #     ...

    def rotate_self(self, angle: float):
        # Function:
        #     Control the end effector of the robot arm to perform a self-rotation movement of angle degrees
        #         at the current position.
        # Input:
        #     angle: Positive floating-point numbers represent clockwise rotation;
        #         negative floating-point numbers represent counterclockwise rotation."
        # Note:
        #     The position of the end effector remains unchanged, the posture is changed,
        #         and the holding state of the gripper remains unchanged.
        # For example:
        #     ...
        #     robot.observe("object")
        #     robot.rotate_self(90)
        #     ...

    def rotate_directionally(self, object_name):
        # Function:
        #     Control the robotic arm's end effector to rotate at the current position, \
        #         aligning it with the object named object_name.
        # Input:
        #     object_name: Name of the object.
        # Note:
        #     The position of the end effector remains unchanged, the posture is changed,
        #         and the holding state of the gripper remains unchanged.
        # For example:
        #     ...
        #     robot.observe()
        #     robot.rotate_directionally("object")
        #     ...

    def home(self):
        # Function:
        #     This method enables the end effector's position and posture of the robot arm to return to the initial pose,
        #         and the gripper is released.
        # Note:
        #     This method is often located at the beginning and end of the code,
        #         used to initialize the robot's position and
        #         have the robot return to the set initial position after the task is completed.
        # For example:
        #     ...
        #     robot.grasp(False)
        #     robot.home()
        #     ...

    def observe(self, object_name):
        # Function:
        #     This method uses the perceptual system to obtain the scene objects' information.
        # Input:
        #     object_name: Name of the object to be observed.
        # Note:
        #     This method is typically placed before the robot executes an action.
        # For example:
        #     ...
        #     robot.observe()
        #     robot.move_horizontally("object")
        #     ...
----------------------------------------------
Based on the method names and the code comments, you must write a markdown-format python program that calls 
these methods to achieve the provided task.
If you don't know how to generate the codes, just say that you don't know, and don't try to make up an answer.
Here is an example:\n
----------------------------------------------
task: Give a water-bottle to Alice.
answer:```python
    robot = M1509()
    robot.home()  # home position
    robot.grasp(False)
    robot.observe("water-bottle")  # obtain the scene infos
    robot.move_horizontally("water-bottle")
    robot.move_vertically("water-bottle")  # move to the cup
    robot.grasp(True)  # grasp the cup
    robot.move_vertically("Alice")
    robot.move_horizontally("Alice")  # move to Alice
    robot.grasp(False)  # release the cup
    robot.home()```
----------------------------------------------\n
{scenario_description}
Attention: You only need to output the code.
Let's begin!
task: {userInput}
answer: 
"""
        self.prompt_template = PromptTemplate(template=self.prompt_template_text,
                                              input_variables=["userInput", "scenario_description"])
        self.code_generator = LLMChain(llm=self.llm, prompt=self.prompt_template, verbose=True)

    def run(self, task: str) -> str:
        code = self.code_generator.run(userInput=running_info['intentions'],
                                       scenario_description=generate_scenario_description("the robot"))
        running_info["code"] = code
        return code


# Behavior Self-correction
class BehaviorSelfCorrection:
    def __init__(self):
        self._llm = OpenAI(model_name=GlobalParam.OpenAI_chat_model,
                           temperature=0.0,
                           max_tokens=GlobalParam.Default_max_tokens)
        prompt_template_text = """
You will play the role of an excellent Python code reviewer. 
You will check the Python robot control codes for a 6 degree-of-freedom robotic arm, DOOSAN M1509 with a gripper. 
Please check the codes based on known facts, the checkpoints. If you check the codes meet the known facts and the
checkpoints, please provide positive comments. On the other hand, if you check the codes doesn't meet the known facts
or the checkpoints, please provide negative comments.
----------------------------------------------
The known facts are that: 
In front of the DOOSAN M1509 robot, {scenario_description}. 
The DOOSAN M1509 robot will be serving Alice.

The checkpoints list as below: 
1. The code must start with 'robot.M1509()'; 
2. When the class initialization is completed and the task execution is finished, 'robot.home()' is needed to ensure that 
the robot returns to its initial position. 
3. Before executing methods that cause changes in the position and posture of the robotic arm's end effector and gripper, 
'robot.observe()' must be used for the acquisition of environment information. 
4. When the robotic arm's end effector and gripper approach an object, horizontal movements should be executed before 
vertical movements. Fuzzy or ambiguous object vocabulary is also acceptable.
5. When the robot transitions from manipulating one object to manipulating another object, the gripper must be released,
using the code, 'robot.grasp(False)'.
6. When a robot completes a grasping and needs to perform a movement action, it must first return to the home position 
like 'robot.home()' after completing the grasping action before proceeding with the movement action. This is done for 
safety reasons.
----------------------------------------------
ATTENTION: Please generate the comments as short as possible, just a single sentence!!!
----------------------------------------------

Let's start!

task: {userTask}
python code: {userPythonCode}
comments: 
"""
        self._prompt_template = PromptTemplate(
            template=prompt_template_text, input_variables=["userTask", "userPythonCode", "scenario_description"])

        self.code_inspector = LLMChain(llm=self._llm, prompt=self._prompt_template, verbose=True)

    def run(self, task_and_code: str) -> str:
        code = running_info["code"]
        task = running_info["intentions"]
        comments = self.code_inspector.run(userTask=task, userPythonCode=code,
                                           scenario_description=generate_scenario_description("the DOOSAN M1509 robot"))
        running_info["comments"] = comments
        return comments


# Execute the code
class ExecuteCodeChain:
    def __init__(self):
        pass

    def run(self, code: str) -> str:
        code_file = running_info["code"]
        # 使用正则表达式提取Python代码
        pattern = r"robot\.[a-zA-Z_]+\([^)]*\)"
        # 使用findall函数提取匹配的代码片段
        matches = re.findall(pattern, code_file)
        # 打印并执行提取的结构化代码
        show_buffer, string_to_print = '', ''
        print("Start to run the code in Robot Coppeliasim...")

        show_buffer += ("Generating robot control codes..." + "<br>")
        show_buffer += ("---------------------------------" + "<br>")
        simxSetIntegerSignal(robot.clientID, "flag", 1, simx_opmode_blocking)
        simxSetStringSignal(robot.clientID, "response", show_buffer, simx_opmode_blocking)

        for i in range(len(matches)):

            code = matches[i]

            show_buffer += (code + "<br>")
            simxSetIntegerSignal(robot.clientID, "flag", 1, simx_opmode_blocking)
            simxSetStringSignal(robot.clientID, "response", show_buffer, simx_opmode_blocking)

            eval(code)  # 执行

            time.sleep(0.1)
            if i == len(matches) - 1:
                show_buffer += ("---------------------------------" + "<br>")
                show_buffer += ("Successfully run the codes in Coppeliasim!" + "<br>")
                simxSetIntegerSignal(robot.clientID, "flag", 1, simx_opmode_blocking)
                simxSetStringSignal(robot.clientID, "response", show_buffer, simx_opmode_blocking)
                return "Successfully run the code in Robot Coppeliasim!"


class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


AnalyzeIntention = AnalyzeHumanIntention()
GenerateCode = GenerateCodeChain()
InspectCode = BehaviorSelfCorrection()
ExecuteCode = ExecuteCodeChain()

tools = [
    Tool(
        name="Analyze-Human-Intention-Tool",
        func=AnalyzeIntention.run,
        description="This tool can help you analyze human intentions and allow you to observe string-format analysis "
                    "results. You just need to input human query into the tool."
    ),
    Tool(
        name="Generate-Code-Tool",
        func=GenerateCode.run,
        description="This tool can help you generate Python-based robot control codes and allow you to observe Python "
                    "codes in markdown format. You just need to input the task into the tool."
    ),
    Tool(
        name="Inspect-Code-Tool",
        func=InspectCode.run,
        description="This tool will help you check whether the codes meet the current situation and comply with the "
                    "specified syntax rules. This tool allows you to observe its generated comments on the codes."
                    "You just need to input the codes into the tool."
    ),
    Tool(
        name="Execute-Code-Tool",
        func=ExecuteCode.run,
        description="This is the only tool you can use to control the physical robot for physical movement."
                    "You can only make the robot in the actual environment perform actions based on codes by using "
                    "this tool. It allows you to observe the final running results of the robot. You just need to input"
                    "the codes into the tool."
    )
]

# Construct a list of tool names for the agent and use it in the prompt words
# to let the agent know more clearly what tools can be used.
tools_list = [tool.name for tool in tools]

template_text = """
Please be an excellent helper and help me, Alice, control a 6 degree-of-freedom robotic arm, called DOOSAN M1509,
with a gripper. The environment of the robotic arm you control is as follows:""" +\
                generate_scenario_description("the robotic arm") + \
"""
Complete my query as best as possible. 
You have access to the following tools:
{tools}
Use the following format:
----------------------------------------------
Question: the input query I must complete
Thought: I should use the tool, "Analyze-Human-Intention-Tool", to think about what to do based on the 'Question'

Action: the action to take, should be "Analyze-Human-Intention-Tool" 
Action Input: The input is the 'Question'.
Observation: the result of the action, should be a string.
Thought: If I observe a clear and specified string, I can proceed to the next tool. Otherwise, 
I need to repeat using this tool until observing a clear and specified task intentions.

Action: the action to take, should be "Generate-Code-Tool" 
Action Input: the task intentions that I observed.
Observation: the codes generated by the tool, "Generate-Code-Tool". I must remember the codes generated.
Thought: If I observe the codes, I can proceed to the next tool. Otherwise, I need to repeat using this tool until
observing the codes.

Action: the action to take, should be "Inspect-Code-Tool"
Action Input: the input to the action, should be the the code I remembered
Observation: the result of the action, should be about the codes.
Thought: If I observe the positive comments, I can proceed to the next tool. Otherwise, I need to repeat using this
tool, "Generate-Code-Tool" with extra input with the comments until observing the positive comments. 
I must remember the codes generated.

Action: the action to take, should be "Execute-Code-Tool" 
Action Input: The input should be the codes with positive comments that I observed.
Observation: I should observe whether the codes are run within the tool, "Execute-Code-Tool".
Thought: If I have observed the the codes running within the tool, I can know the final answer based on the 
'Observation'. Otherwise, I need to repeat using this tool until observing the codes running within the tool.

Final Answer: the final answer to the original input question. I must place the phrase "Final Answer" at the beginning.
----------------------------------------------
Please use the above format!
Begin!

Question: {input}
{agent_scratchpad}"""

# Instantiate prompts and encapsulate an interface that can input
prompt = CustomPromptTemplate(
    template=template_text,
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)


output_parser = CustomOutputParser()


llm_outside_chain = LLMChain(llm=OpenAI(model_name=GlobalParam.OpenAI_chat_model,
                                        temperature=GlobalParam.Default_temperature,
                                        max_tokens=GlobalParam.Default_max_tokens),
                             prompt=prompt)


agent = LLMSingleActionAgent(
    llm_chain=llm_outside_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tools_list,
    handle_parsing_errors=True,
)


last_query = None

simxSetStringSignal(robot.clientID, "python", "start", simx_opmode_blocking)

while True:

    _, send = simxGetIntegerSignal(robot.clientID, "send", simx_opmode_blocking)

    if send == 1:

        _, query = simxGetStringSignal(robot.clientID, "query", simx_opmode_blocking)
        query = query.decode('utf-8')

        if query != "":

            if query == "exit":
                simxSetStringSignal(robot.clientID, "llm", "", simx_opmode_blocking)
                simxSetStringSignal(robot.clientID, "query", "", simx_opmode_blocking)
                simxSetStringSignal(robot.clientID, "python", "close", simx_opmode_blocking)
                exit()

            if last_query != query:

                agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
                with get_openai_callback() as cb:
                    agent_executor.run(query)
                del agent_executor

                running_info["total_tokens"] = cb.total_tokens
                running_info["total_cost"] = cb.total_cost
                pprint.pprint(running_info)

                with open("RunningInfos/" + datetime.datetime.now().strftime("%Y%m%d%H%M") + ".json", "w") as f:
                    json.dump(running_info, f, indent=4)
                    f.close()

                last_query = query
            else:
                simxSetIntegerSignal(robot.clientID, "flag", 0, simx_opmode_blocking)
                pass

            query = ""
        else:
            print("Query is empty! ...")
    else:
        print("No message! ...")
