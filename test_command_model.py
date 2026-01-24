"""Test the enhanced Command model implementation."""
from dataclasses import dataclass
from typing import Callable, Optional, Any, List

@dataclass
class Command:
    """Command configuration with method and optional CLI argument mapping."""
    method: Callable[[], bool]
    help_text: str
    cli_arg: Optional[str] = None
    cli_action: Optional[str] = None
    cli_type: Optional[type] = None
    cli_default: Optional[Any] = None
    cli_help: Optional[str] = None
    aliases: Optional[List[str]] = None

# Test command creation
def test_quit():
    print("Quitting...")
    return True

def test_help():
    print("Showing help...")
    return False

# Create command instances
quit_cmd = Command(
    method=test_quit,
    help_text='Exit the application',
    aliases=['exit', 'bye']
)

help_cmd = Command(
    method=test_help,
    help_text='Show all available commands'
)

# Test command registry
commands = {
    'quit': quit_cmd,
    'help': help_cmd,
}

# Expand aliases
expanded = {}
for name, cmd in commands.items():
    expanded[name] = cmd
    if cmd.aliases:
        for alias in cmd.aliases:
            expanded[alias] = cmd

print("Command Registry Test")
print(f"\nRegistered commands: {list(expanded.keys())}")
print(f"\nTotal entries (with aliases): {len(expanded)}")

# Test invocation
print("\n\nTesting command invocation:")
for cmd_name in ['quit', 'exit', 'help']:
    if cmd_name in expanded:
        print(f"\nInvoking '{cmd_name}':")
        result = expanded[cmd_name].method()
        print(f"  Result: {result}")
        print(f"  Help: {expanded[cmd_name].help_text}")
        if expanded[cmd_name].aliases:
            print(f"  Aliases: {', '.join(expanded[cmd_name].aliases)}")

print("\n" + "=" * 60)
print(" Command model test complete!")
