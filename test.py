#!/usr/bin/env python

"""
A simple test file with a few code blocks.
"""

# This is a simple function
def hello(name):
    """Says hello to someone."""
    return f"Hello, {name}!"

# This is a simple class
class Person:
    """Represents a person."""
    
    def __init__(self, name, age):
        """Initialize a new person."""
        self.name = name
        self.age = age
    
    def greet(self):
        """Greet the person."""
        return f"Hello, {self.name}!"
    
    def is_adult(self):
        """Check if the person is an adult."""
        return self.age >= 18

# Test code
if __name__ == "__main__":
    # Create a new person
    person = Person("Alice", 30)
    
    # Greet the person
    print(person.greet())
    
    # Check if the person is an adult
    print(f"Is {person.name} an adult? {person.is_adult()}") 