import json
from typing import List, Optional, TypedDict, Union

from lib.llm import LLM
from lib.memory import ShortTermMemory
from lib.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, UserMessage
from lib.state_machine import EntryPoint, Run, StateMachine, Step, Termination
from lib.tooling import Tool, ToolCall


# Define the state schema
class AgentState(TypedDict):
    user_query: str  # The current user query being processed
    instructions: str  # System instructions for the agent
    messages: List[BaseMessage]  # List of conversation messages
    current_tool_calls: Optional[List[ToolCall]]  # Current pending tool calls
    total_tokens: int  # Track the cumulative total


class Agent:
    def __init__(
        self,
        instructions: str,
        tools: List[Tool] = [],
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.7,
    ):
        """
        Initialize an Agent

        Args:
            model_name: Name/identifier of the LLM model to use
            instructions: System instructions for the agent
            tools: Optional list of tools available to the agent
            temperature: Temperature parameter for LLM (default: 0.7)
        """
        self.instructions = instructions
        self.tools = tools if tools else []
        self.model_name = model_name
        self.temperature = temperature

        # Initialize memory and state machine
        self.memory = ShortTermMemory()
        self.workflow = self._create_state_machine()

    def _prepare_messages_step(self, state: AgentState) -> AgentState:
        """Step logic: Prepare messages for LLM consumption"""
        messages: List[BaseMessage] = list(state.get("messages", []) or [])

        # If no messages exist, start with system message
        if not messages:
            messages.append(SystemMessage(content=state["instructions"]))

        # Add the new user message
        messages.append(UserMessage(content=state["user_query"]))

        return {"messages": messages, "session_id": state["session_id"]}

    def _llm_step(self, state: AgentState) -> AgentState:
        """Step logic: Process the current state through the LLM"""
        # Initialize LLM
        llm = LLM(model=self.model_name, temperature=self.temperature, tools=self.tools)

        response = llm.invoke(state["messages"])
        tool_calls = response.tool_calls if response.tool_calls else None

        current_total = state.get("total_tokens", 0)
        if response.token_usage:
            current_total += response.token_usage.total_tokens

        # Create AI message with content and tool calls
        ai_message = AIMessage(
            content=response.content,
            tool_calls=tool_calls,
        )

        return {
            "messages": state["messages"] + [ai_message],
            "current_tool_calls": tool_calls,
            "session_id": state["session_id"],
            "total_tokens": current_total,
        }

    def _tool_step(self, state: AgentState) -> AgentState:
        """Step logic: Execute any pending tool calls"""
        tool_calls = state["current_tool_calls"] or []
        tool_messages = []

        for call in tool_calls:
            # Access tool call data correctly
            function_name = call.function.name
            function_args = json.loads(call.function.arguments)
            tool_call_id = call.id
            # Find the matching tool
            tool = next((t for t in self.tools if t.name == function_name), None)
            if tool:
                result = str(tool(**function_args))
                tool_message = ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                tool_messages.append(tool_message)

        # Clear tool calls and add results to messages
        return {
            "messages": state["messages"] + tool_messages,
            "current_tool_calls": None,
            "session_id": state["session_id"],
        }

    def _create_state_machine(self) -> StateMachine[AgentState]:
        """Create the internal state machine for the agent"""
        machine = StateMachine[AgentState](AgentState)

        # Create steps
        entry = EntryPoint[AgentState]()
        message_prep = Step[AgentState]("message_prep", self._prepare_messages_step)
        llm_processor = Step[AgentState]("llm_processor", self._llm_step)
        tool_executor = Step[AgentState]("tool_executor", self._tool_step)
        termination = Termination[AgentState]()

        machine.add_steps(
            [entry, message_prep, llm_processor, tool_executor, termination]
        )

        # Add transitions
        machine.connect(entry, message_prep)
        machine.connect(message_prep, llm_processor)

        # Transition based on whether there are tool calls
        def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
            """Transition logic: Check if there are tool calls"""
            if state.get("current_tool_calls"):
                return tool_executor
            return termination

        machine.connect(llm_processor, [tool_executor, termination], check_tool_calls)
        machine.connect(
            tool_executor, llm_processor
        )  # Go back to llm after tool execution

        return machine

    def invoke(self, query: str, session_id: Optional[str] = None) -> Run:
        """
        Run the agent on a query

        Args:
            query: The user's query to process
            session_id: Optional session identifier (uses "default" if None)

        Returns:
            The final run object after processing
        """
        session_id = session_id or "default"

        # Create session if it doesn't exist
        self.memory.create_session(session_id)

        # Get previous messages from last run if available
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)
        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state["messages"]

        initial_state: AgentState = {
            "user_query": query,
            "instructions": self.instructions,
            "messages": previous_messages,
            "current_tool_calls": None,
            "session_id": session_id,
        }

        run_object = self.workflow.run(initial_state)

        # Store the complete run object in memory
        self.memory.add(run_object, session_id)

        return run_object

    def get_session_runs(self, session_id: Optional[str] = None) -> List[Run]:
        """Get all Run objects for a session

        Args:
            session_id: Optional session ID (uses "default" if None)

        Returns:
            List of Run objects in the session
        """
        return self.memory.get_all_objects(session_id)

    def reset_session(self, session_id: Optional[str] = None):
        """Reset memory for a specific session

        Args:
            session_id: Optional session to reset (uses "default" if None)
        """
        self.memory.reset(session_id)

    def pretty_print_memory(self, session_id: Optional[str] = None) -> None:
        """Pretty-print the agent's memory for the most recent run in a session.

        Uses ANSI colours and emoji to make each message type easy to scan:
          - ⚙️  System (gray)
          - 👤 User (blue)
          - 🤖 Assistant (green)
          - 🔧 Tool call (yellow)
          - 📦 Tool response (magenta)

        Args:
            session_id: Optional session ID (uses "default" if None)
        """
        # ANSI escape codes
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        GRAY = "\033[90m"
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"

        session_id = session_id or "default"
        last_run: Optional[Run] = self.memory.get_last_object(session_id)
        if last_run is None:
            print(f"{DIM}(no runs recorded for session '{session_id}'){RESET}")
            return

        final_state = last_run.get_final_state()
        if not final_state or not final_state.get("messages"):
            print(f"{DIM}(no messages recorded in last run){RESET}")
            return

        messages: List[BaseMessage] = final_state["messages"]
        total_tokens = final_state.get("total_tokens", 0)

        header = f"🧵 Agent Memory — session='{session_id}' run={last_run.run_id[:8]} ({len(messages)} messages)"
        print(f"\n{BOLD}{CYAN}{header}{RESET}")
        print(f"{CYAN}{'─' * min(len(header), 80)}{RESET}")

        def _indent(text: Optional[str], prefix: str = "   ") -> str:
            if text is None:
                return ""
            return "\n".join(prefix + line for line in str(text).splitlines())

        for idx, msg in enumerate(messages, start=1):
            if isinstance(msg, SystemMessage):
                print(f"{GRAY}{BOLD}[{idx}] ⚙️  SYSTEM{RESET}")
                print(f"{GRAY}{_indent(msg.content)}{RESET}")

            elif isinstance(msg, UserMessage):
                print(f"{BLUE}{BOLD}[{idx}] 👤 USER{RESET}")
                print(f"{BLUE}{_indent(msg.content)}{RESET}")

            elif isinstance(msg, AIMessage):
                label = f"{GREEN}{BOLD}[{idx}] 🤖 ASSISTANT{RESET}"
                if msg.token_usage:
                    label += (
                        f" {DIM}(tokens: prompt={msg.token_usage.prompt_tokens} "
                        f"completion={msg.token_usage.completion_tokens} "
                        f"total={msg.token_usage.total_tokens}){RESET}"
                    )
                print(label)
                if msg.content:
                    print(f"{GREEN}{_indent(msg.content)}{RESET}")
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = tc.function.arguments
                        try:
                            fn_args = json.dumps(json.loads(fn_args), indent=2)
                        except (ValueError, TypeError):
                            pass
                        print(
                            f"{YELLOW}   🔧 tool_call {tc.id}: "
                            f"{BOLD}{fn_name}{RESET}{YELLOW}({RESET}"
                        )
                        print(f"{YELLOW}{_indent(fn_args, '      ')}{RESET}")
                        print(f"{YELLOW}   ){RESET}")

            elif isinstance(msg, ToolMessage):
                print(
                    f"{MAGENTA}{BOLD}[{idx}] 📦 TOOL RESULT{RESET}"
                    f" {DIM}name={msg.name} call_id={msg.tool_call_id}{RESET}"
                )
                content = msg.content
                # Try to prettify JSON tool output
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, str):
                        try:
                            parsed = json.loads(parsed)
                        except (ValueError, TypeError):
                            pass
                    content = json.dumps(parsed, indent=2, ensure_ascii=False)
                except (ValueError, TypeError):
                    pass
                print(f"{MAGENTA}{_indent(content)}{RESET}")

            else:
                # Fallback for any unexpected message type
                print(f"{BOLD}[{idx}] ❓ {type(msg).__name__}{RESET}")
                print(_indent(repr(msg)))

        if total_tokens:
            print(
                f"{CYAN}{'─' * 40}{RESET}\n"
                f"{CYAN}{BOLD}📊 Total tokens used in this run: {total_tokens}{RESET}"
            )
        print()
