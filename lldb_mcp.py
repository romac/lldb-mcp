#!/usr/bin/env python3
import os
import re
import sys
import uuid
import asyncio
import subprocess
import pty
import fcntl
import termios
import struct
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context

# Class to handle LLDB sessions
class LldbSession:
    def __init__(self, session_id: str, lldb_path: str, working_dir: Optional[str] = None):
        self.id = session_id
        self.lldb_path = lldb_path
        self.working_dir = working_dir or os.getcwd()
        self.process = None
        self.master_fd = None
        self.slave_fd = None
        self.target = None
        self.ready = False
    
    async def start(self) -> str:
        """Start the LLDB process with a PTY"""
        print(f"Starting LLDB process with path: {self.lldb_path}")
        
        # Create a pseudo-terminal pair
        self.master_fd, self.slave_fd = pty.openpty()
        print(f"Created PTY pair: master={self.master_fd}, slave={self.slave_fd}")
        
        # Make the terminal raw
        old_settings = termios.tcgetattr(self.slave_fd)
        new_settings = termios.tcgetattr(self.slave_fd)
        new_settings[3] = new_settings[3] & ~termios.ECHO  # Disable echo
        termios.tcsetattr(self.slave_fd, termios.TCSADRAIN, new_settings)
        
        # Start LLDB process
        cmd = [self.lldb_path]
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            cwd=self.working_dir
        )
        print(f"LLDB process created with PID: {self.process.pid}")
        
        # Close slave end in parent process
        os.close(self.slave_fd)
        self.slave_fd = None
        
        # Make the master fd non-blocking
        flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        
        # Wait for initial prompt
        print("Waiting for initial prompt...")
        output = await self.read_until_prompt()
        print("Initial prompt received")

        self.ready = True
        
        # Send version command to confirm LLDB is working
        print("Sending version command")
        version_output = await self.execute_command("version")
        output += version_output
        
        return output
    
    async def execute_command(self, command: str) -> str:
        """Execute an LLDB command and return the output"""
        if not self.ready:
            raise RuntimeError("LLDB session is not ready")

        if not self.process:
            raise RuntimeError("LLDB session is not ready: no process")

        if self.process.returncode is not None:
            raise RuntimeError("LLDB session is not ready: process has terminated, code: %d" % self.process.returncode)
        
        # Write command to master fd
        print(f"Executing command: {command}")
        os.write(self.master_fd, f"{command}\n".encode())
        
        # Read response until prompt
        return await self.read_until_prompt()
    
    async def read_until_prompt(self) -> str:
        """Read from LLDB until prompt is encountered"""
        if not self.master_fd:
            raise RuntimeError("PTY not initialized")
            
        buffer = b""
        prompt_pattern = b"(lldb)"
        
        start_time = asyncio.get_event_loop().time()
        print("Starting to read until prompt")
        
        # Read until we see the prompt or timeout
        while True:
            # Check for timeout
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > 10.0:
                print(f"Global timeout reached after {current_time - start_time:.1f} seconds")
                return buffer.decode('utf-8', errors='replace') + "\n[Timeout waiting for LLDB response]"
            
            # Check if process has terminated
            if self.process and self.process.returncode is not None:
                print(f"Process terminated with code: {self.process.returncode}")
                if buffer:
                    return buffer.decode('utf-8', errors='replace')
                raise RuntimeError(f"LLDB process terminated with code {self.process.returncode}")
            
            try:
                # Try to read from the master fd
                chunk = os.read(self.master_fd, 1024)
                if chunk:
                    print(f"Read {len(chunk)} bytes from PTY")
                    buffer += chunk
                    
                    # Print readable content for debugging
                    try:
                        decoded = chunk.decode('utf-8', errors='replace')
                        print(f"Read data: {decoded.strip()}")
                    except:
                        pass
                    
                    # Check if buffer contains the prompt
                    if prompt_pattern in buffer:
                        print("Found LLDB prompt in buffer")
                        return buffer.decode('utf-8', errors='replace')
            except BlockingIOError:
                # No data available yet, yield control
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error reading from PTY: {str(e)}")
                if buffer:
                    return buffer.decode('utf-8', errors='replace') + f"\n[Error reading from LLDB: {str(e)}]"
                raise RuntimeError(f"Error reading from LLDB: {str(e)}")
    
    async def cleanup(self):
        """Clean up LLDB resources"""
        print("Cleaning up LLDB session")
        try:
            if self.master_fd is not None:
                # Send quit command
                try:
                    os.write(self.master_fd, b"quit\n")
                    # Wait briefly for the process to exit
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Error sending quit command: {e}")
                
            if self.process and self.process.returncode is None:
                print("Terminating LLDB process")
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 2.0)
                except asyncio.TimeoutError:
                    print("Force killing LLDB process")
                    self.process.kill()
                    await self.process.wait()
                    
            # Close the PTY master fd
            if self.master_fd is not None:
                print(f"Closing master fd: {self.master_fd}")
                os.close(self.master_fd)
                self.master_fd = None
                
            # Close the PTY slave fd if it's still open
            if self.slave_fd is not None:
                print(f"Closing slave fd: {self.slave_fd}")
                os.close(self.slave_fd)
                self.slave_fd = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.process = None
            self.ready = False
            print("LLDB session cleanup completed")


@dataclass
class AppContext:
    """Application context for storing active sessions"""
    sessions: Dict[str, LldbSession]


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with shared session state"""
    # Initialize on startup
    sessions = {}
    try:
        yield AppContext(sessions=sessions)
    finally:
        # Cleanup all sessions on shutdown
        for session_id, session in list(sessions.items()):
            await session.cleanup()
            sessions.pop(session_id, None)


# Create an MCP server
mcp = FastMCP("lldb-mcp", lifespan=app_lifespan)


# Helper function to get session from context
def get_session(ctx: Context, session_id: str) -> LldbSession:
    """Get a session by ID or raise an error if not found"""
    # Access the lifespan context through request_context
    sessions = ctx.request_context.lifespan_context.sessions
    if session_id not in sessions:
        raise ValueError(f"No active LLDB session with ID: {session_id}")
    return sessions[session_id]


@mcp.tool()
async def lldb_start(ctx: Context, lldb_path: str = "lldb", working_dir: str = None) -> str:
    """Start a new LLDB session"""
    session_id = str(uuid.uuid4())
    print(f"Starting new LLDB session with ID: {session_id}")
    
    try:
        # Verify lldb path
        try:
            print(f"Verifying lldb path: {lldb_path}")
            proc = await asyncio.create_subprocess_exec(
                lldb_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            print(f"Communicating with lldb")
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
                print(f"Received response from lldb")
                if proc.returncode != 0:
                    error_msg = stderr.decode('utf-8', errors='replace').strip()
                    return f"Failed to start LLDB: Invalid lldb path '{lldb_path}'. Error: {error_msg}"
            except asyncio.TimeoutError:
                print("Timeout while verifying lldb path")
                proc.kill()
                await proc.wait()
                return f"Failed to start LLDB: Timeout while verifying lldb path '{lldb_path}'"
        except Exception as e:
            print(f"Error verifying lldb path: {str(e)}")
            return f"Failed to start LLDB: Invalid lldb path '{lldb_path}'. Error: {str(e)}"
        
        # Use provided working directory or current dir
        working_dir = working_dir or os.getcwd()
        print(f"Using working directory: {working_dir}")
        
        # Create new LLDB session
        session = LldbSession(session_id=session_id, lldb_path=lldb_path, working_dir=working_dir)
        print(f"Created new LLDB session")
        
        # Start LLDB process with timeout
        try:
            print("Starting LLDB process...")
            output = await asyncio.wait_for(session.start(), timeout=10.0)
            print(f"LLDB session started successfully")
        except asyncio.TimeoutError:
            print("Timeout while starting LLDB session")
            await session.cleanup()
            return f"Failed to start LLDB: Timeout while initializing LLDB session"
        except Exception as e:
            print(f"Error starting LLDB session: {str(e)}")
            await session.cleanup()
            return f"Failed to start LLDB: {str(e)}"
        
        # Store session in context
        ctx.request_context.lifespan_context.sessions[session_id] = session
        print(f"Stored session in context")
        return f"LLDB session started with ID: {session_id}\n\nOutput:\n{output}"
    
    except Exception as e:
        print(f"Unexpected error in lldb_start: {str(e)}")
        return f"Failed to start LLDB: {str(e)}"


@mcp.tool()
async def lldb_load(ctx: Context, session_id: str, program: str, arguments: List[str] = None) -> str:
    """Load a program into LLDB"""
    try:
        session = get_session(ctx, session_id)
        
        # Normalize path if working directory is set
        if session.working_dir and not os.path.isabs(program):
            program = os.path.join(session.working_dir, program)
        
        # Create target
        output = await session.execute_command(f"file \"{program}\"")
        
        # Set program arguments if provided
        if arguments:
            args_str = " ".join(f'"{arg}"' for arg in arguments)
            args_output = await session.execute_command(f"settings set -- target.run-args {args_str}")
            output += f"\n{args_output}"
        
        # Update session target
        session.target = program
        
        return f"Program loaded: {program}\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to load program: {str(e)}"


@mcp.tool()
async def lldb_command(ctx: Context, session_id: str, command: str) -> str:
    """Execute an LLDB command"""
    try:
        session = get_session(ctx, session_id)
        output = await session.execute_command(command)
        return f"Command: {command}\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to execute command: {str(e)}"


@mcp.tool()
async def lldb_terminate(ctx: Context, session_id: str) -> str:
    """Terminate an LLDB session"""
    try:
        session = get_session(ctx, session_id)
        await session.cleanup()
        ctx.request_context.lifespan_context.sessions.pop(session_id, None)
        return f"LLDB session terminated: {session_id}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to terminate LLDB session: {str(e)}"


@mcp.tool()
def lldb_list_sessions(ctx: Context) -> str:
    """List all active LLDB sessions"""
    sessions = ctx.request_context.lifespan_context.sessions
    session_info = []
    
    for session_id, session in sessions.items():
        session_info.append({
            "id": session_id,
            "target": session.target or "No program loaded",
            "working_dir": session.working_dir
        })
    
    return f"Active LLDB Sessions ({len(sessions)}):\n\n{session_info}"


@mcp.tool()
async def lldb_attach(ctx: Context, session_id: str, pid: int) -> str:
    """Attach to a running process"""
    try:
        session = get_session(ctx, session_id)
        output = await session.execute_command(f"process attach -p {pid}")
        return f"Attached to process {pid}\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to attach to process: {str(e)}"


@mcp.tool()
async def lldb_load_core(ctx: Context, session_id: str, program: str, core_path: str) -> str:
    """Load a core dump file"""
    try:
        session = get_session(ctx, session_id)
        
        # First load the program
        file_output = await session.execute_command(f"file \"{program}\"")
        
        # Then load the core file
        core_output = await session.execute_command(f"target core \"{core_path}\"")
        
        # Get backtrace to show initial state
        backtrace_output = await session.execute_command("bt")
        
        return f"Core file loaded: {core_path}\n\nOutput:\n{file_output}\n{core_output}\n\nBacktrace:\n{backtrace_output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to load core file: {str(e)}"


@mcp.tool()
async def lldb_set_breakpoint(ctx: Context, session_id: str, location: str, condition: str = None) -> str:
    """Set a breakpoint"""
    try:
        session = get_session(ctx, session_id)
        
        # Set breakpoint
        output = await session.execute_command(f"breakpoint set --name \"{location}\"")
        
        # Set condition if provided
        if condition:
            # Extract breakpoint number from output (example: "Breakpoint 1: where = ...")
            match = re.search(r"Breakpoint (\d+):", output)
            if match:
                bp_num = match.group(1)
                condition_output = await session.execute_command(f"breakpoint modify -c \"{condition}\" {bp_num}")
                output += f"\n{condition_output}"
        
        return f"Breakpoint set at: {location}{' with condition: ' + condition if condition else ''}\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to set breakpoint: {str(e)}"


@mcp.tool()
async def lldb_continue(ctx: Context, session_id: str) -> str:
    """Continue program execution"""
    try:
        session = get_session(ctx, session_id)
        output = await session.execute_command("continue")
        return f"Continued execution\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to continue execution: {str(e)}"


@mcp.tool()
async def lldb_step(ctx: Context, session_id: str, instructions: bool = False) -> str:
    """Step program execution"""
    try:
        session = get_session(ctx, session_id)
        command = "si" if instructions else "s"  # step instruction vs. step
        output = await session.execute_command(command)
        return f"Stepped {instructions and 'instruction' or 'line'}\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to step: {str(e)}"


@mcp.tool()
async def lldb_next(ctx: Context, session_id: str, instructions: bool = False) -> str:
    """Step over function calls"""
    try:
        session = get_session(ctx, session_id)
        command = "ni" if instructions else "n"  # next instruction vs. next
        output = await session.execute_command(command)
        return f"Stepped over {instructions and 'instruction' or 'function call'}\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to step over: {str(e)}"


@mcp.tool()
async def lldb_finish(ctx: Context, session_id: str) -> str:
    """Execute until the current function returns"""
    try:
        session = get_session(ctx, session_id)
        output = await session.execute_command("finish")
        return f"Finished current function\n\nOutput:\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to finish function: {str(e)}"


@mcp.tool()
async def lldb_backtrace(ctx: Context, session_id: str, full: bool = False, limit: int = None) -> str:
    """Show call stack"""
    try:
        session = get_session(ctx, session_id)
        
        # Build backtrace command with options
        command = "bt"
        if full:
            command += " all"  # Show all frame variables
        if limit is not None:
            command += f" -c {limit}"  # Frame count limit
        
        output = await session.execute_command(command)
        return f"Backtrace{' (full)' if full else ''}{f' (limit: {limit})' if limit else ''}:\n\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to get backtrace: {str(e)}"


@mcp.tool()
async def lldb_print(ctx: Context, session_id: str, expression: str) -> str:
    """Print value of expression"""
    try:
        session = get_session(ctx, session_id)
        output = await session.execute_command(f"p {expression}")
        return f"Print {expression}:\n\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to print expression: {str(e)}"

@mcp.tool()
async def lldb_examine(ctx: Context, session_id: str, expression: str, format: str = "x", count: int = 1) -> str:
    """Examine memory"""
    try:
        session = get_session(ctx, session_id)
        
        # Map format codes to LLDB format specifiers
        format_map = {
            "x": "x",      # hex
            "d": "d",      # decimal
            "u": "u",      # unsigned decimal
            "o": "o",      # octal
            "t": "t",      # binary
            "i": "i",      # instruction
            "c": "c",      # character
            "f": "f",      # float
            "s": "s"       # string
        }
        
        # Get LLDB format or default to hex
        lldb_format = format_map.get(format, "x")
        
        # Build memory examine command
        command = f"memory read -f {lldb_format} -c {count} {expression}"
        output = await session.execute_command(command)
        
        return f"Examine {expression} (format: {format}, count: {count}):\n\n{output}"
    
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Failed to examine memory: {str(e)}"


@mcp.tool()
async def lldb_info_registers(ctx: Context, session_id: str, register: str = None) -> str:
   """Display registers"""
   try:
       session = get_session(ctx, session_id)
       
       # Build register info command
       command = "register read"
       if register:
           command += f" {register}"
       
       output = await session.execute_command(command)
       return f"Register info{f' for {register}' if register else ''}:\n\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to get register info: {str(e)}"


@mcp.tool()
async def lldb_watchpoint(ctx: Context, session_id: str, expression: str, watch_type: str = "write") -> str:
   """Set a watchpoint on a variable or memory address"""
   try:
       session = get_session(ctx, session_id)
       
       # Map watch types to LLDB options
       watch_options = {
           "read": "r",
           "write": "w",
           "read_write": "rw"
       }
       
       option = watch_options.get(watch_type, "w")
       
       output = await session.execute_command(f"watchpoint set expression -- {expression} -w {option}")
       return f"Watchpoint set on {expression} (type: {watch_type})\n\nOutput:\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to set watchpoint: {str(e)}"


@mcp.tool()
async def lldb_frame_info(ctx: Context, session_id: str, frame_index: int = 0) -> str:
   """Get detailed information about a stack frame"""
   try:
       session = get_session(ctx, session_id)
       
       # First select the frame
       frame_output = await session.execute_command(f"frame select {frame_index}")
       
       # Get frame variables
       vars_output = await session.execute_command("frame variable")
       
       # Get frame source info
       source_output = await session.execute_command("source list")
       
       return f"Frame {frame_index} info:\n\n{frame_output}\n\nVariables:\n{vars_output}\n\nSource:\n{source_output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to get frame info: {str(e)}"


@mcp.tool()
async def lldb_run(ctx: Context, session_id: str) -> str:
   """Run the loaded program"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command("run")
       return f"Running program\n\nOutput:\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to run program: {str(e)}"


@mcp.tool()
async def lldb_kill(ctx: Context, session_id: str) -> str:
   """Kill the running process"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command("process kill")
       return f"Killed process\n\nOutput:\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to kill process: {str(e)}"


@mcp.tool()
async def lldb_thread_list(ctx: Context, session_id: str) -> str:
   """List all threads in the current process"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command("thread list")
       return f"Thread list:\n\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to list threads: {str(e)}"


@mcp.tool()
async def lldb_thread_select(ctx: Context, session_id: str, thread_id: int) -> str:
   """Select a specific thread"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command(f"thread select {thread_id}")
       
       # Get backtrace for the selected thread
       bt_output = await session.execute_command("bt")
       
       return f"Selected thread {thread_id}\n\nOutput:\n{output}\n\nBacktrace:\n{bt_output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to select thread: {str(e)}"


@mcp.tool()
async def lldb_breakpoint_list(ctx: Context, session_id: str) -> str:
   """List all breakpoints"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command("breakpoint list")
       return f"Breakpoint list:\n\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to list breakpoints: {str(e)}"


@mcp.tool()
async def lldb_breakpoint_delete(ctx: Context, session_id: str, breakpoint_id: int) -> str:
   """Delete a breakpoint"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command(f"breakpoint delete {breakpoint_id}")
       return f"Deleted breakpoint {breakpoint_id}\n\nOutput:\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to delete breakpoint: {str(e)}"


@mcp.tool()
async def lldb_expression(ctx: Context, session_id: str, expression: str) -> str:
   """Evaluate an expression in the current frame"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command(f"expression -- {expression}")
       return f"Expression evaluation: {expression}\n\nOutput:\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to evaluate expression: {str(e)}"


@mcp.tool()
async def lldb_process_info(ctx: Context, session_id: str) -> str:
   """Get information about the current process"""
   try:
       session = get_session(ctx, session_id)
       output = await session.execute_command("process status")
       
       # Get additional process info
       pid_output = await session.execute_command("process info")
       
       return f"Process information:\n\n{output}\n\nDetails:\n{pid_output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to get process info: {str(e)}"


@mcp.tool()
async def lldb_disassemble(ctx: Context, session_id: str, location: str = None, count: int = 10) -> str:
   """Disassemble code"""
   try:
       session = get_session(ctx, session_id)
       
       # Build disassemble command
       command = "disassemble"
       if location:
           command += f" --name {location}"
       command += f" -c {count}"
       
       output = await session.execute_command(command)
       return f"Disassembly{f' of {location}' if location else ''}:\n\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to disassemble: {str(e)}"


@mcp.tool()
async def lldb_help(ctx: Context, session_id: str, command: str = None) -> str:
   """Get help for LLDB commands"""
   try:
       session = get_session(ctx, session_id)

       print(f"Getting help for command: {command}")
       
       if command:
           output = await session.execute_command(f"help {command}")
           return f"Help for '{command}':\n\n{output}"
       else:
           output = await session.execute_command("help")
           return f"LLDB help overview:\n\n{output}"
   
   except ValueError as e:
       return str(e)
   except Exception as e:
       return f"Failed to get help: {str(e)}"


if __name__ == "__main__":
   try:
       mcp.run()
   except KeyboardInterrupt:
       print("LLDB-MCP server stopped")
