"""Core module for Liquid Neural Networks."""


def greet(name: str = "World") -> str:
    """Return a greeting message.
    
    Args:
        name: Name to greet
        
    Returns:
        Greeting message
    """
    return f"Hello, {name}! Welcome to Liquid Neural Networks."


def main() -> None:
    """Main entry point."""
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "World"
    print(greet(name))


if __name__ == "__main__":
    main()