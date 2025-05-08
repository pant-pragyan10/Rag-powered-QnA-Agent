"""
Tools for the agent to use when processing queries.
"""
import math
import requests
from typing import Dict, Any, List, Optional

class CalculatorTool:
    """Tool for performing basic calculations."""
    
    def __init__(self):
        self.name = "calculator"
        self.description = "Useful for performing mathematical calculations"
    
    def run(self, expression: str) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation
        """
        try:
            # Special case for age calculation
            import re
            import datetime
            import math
            
            # Get current year
            current_year = datetime.datetime.now().year
            
            # Check if this is an age calculation
            age_match = re.search(r'born in (\d{4})', expression.lower())
            if age_match:
                birth_year = int(age_match.group(1))
                age = current_year - birth_year
                return {
                    "tool": self.name,
                    "input": expression,
                    "output": f"If you were born in {birth_year}, you would be approximately {age} years old in {current_year}."
                }
                
            # Check if this is a square root calculation
            sqrt_match = re.search(r'square root of (\d+(\.\d+)?)|sqrt\s*\(?\s*(\d+(\.\d+)?)\s*\)?', expression.lower())
            if sqrt_match:
                # Extract the number from whichever group matched
                number = sqrt_match.group(1) or sqrt_match.group(3)
                if number:
                    num_value = float(number)
                    result = math.sqrt(num_value)
                    return {
                        "tool": self.name,
                        "input": expression,
                        "output": f"The square root of {num_value} is {result:.6f}"
                    }
            
            # Check if this is likely a non-mathematical query
            # If it contains too many words and not enough numbers/operators, it's probably not a calculation
            words = re.findall(r'\b[a-zA-Z]{3,}\b', expression.lower())
            if len(words) > 3 and not re.search(r'[0-9][\+\-\*\/][0-9]', expression):
                return {
                    "tool": self.name,
                    "input": expression,
                    "output": "This doesn't appear to be a mathematical calculation. Please try a different query."
                }
            
            # Extract just the mathematical expression if there's text around it
            # Look for patterns like numbers, operators, and math functions
            math_pattern = r'[\d\+\-\*\/\(\)\^\s\.]+'
            math_matches = re.findall(math_pattern, expression)
            if math_matches:
                # Join all the matches to form a clean expression
                clean_expression = ''.join(math_matches).strip()
                if clean_expression:
                    expression = clean_expression
            
            # Verify that we have a valid mathematical expression
            if not re.search(r'\d', expression):
                return {
                    "tool": self.name,
                    "input": expression,
                    "output": "No valid mathematical expression found in the input."
                }
            
            # Use safer eval with limited scope
            allowed_names = {
                "abs": abs,
                "max": max,
                "min": min,
                "pow": pow,
                "round": round,
                "sum": sum,
                "math": math,
                "sqrt": math.sqrt
            }
            
            # Clean the expression
            expression = expression.replace('^', '**')
            
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            return {
                "tool": self.name,
                "input": expression,
                "output": str(result)
            }
        except Exception as e:
            return {
                "tool": self.name,
                "input": expression,
                "output": f"Error: {str(e)}"
            }


class DictionaryTool:
    """Tool for looking up word definitions."""
    
    def __init__(self):
        self.name = "dictionary"
        self.description = "Useful for looking up the definition of words"
        self.api_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
    
    def run(self, word: str) -> Dict[str, Any]:
        """
        Look up the definition of a word.
        
        Args:
            word: Word to look up
            
        Returns:
            Definition of the word
        """
        try:
            # Clean the word
            word = word.strip().lower()
            
            # Make API request
            response = requests.get(f"{self.api_url}{word}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract the first definition
                if data and isinstance(data, list) and len(data) > 0:
                    meanings = data[0].get("meanings", [])
                    if meanings and len(meanings) > 0:
                        definitions = meanings[0].get("definitions", [])
                        if definitions and len(definitions) > 0:
                            definition = definitions[0].get("definition", "No definition found")
                            return {
                                "tool": self.name,
                                "input": word,
                                "output": definition
                            }
            
            return {
                "tool": self.name,
                "input": word,
                "output": "No definition found"
            }
        except Exception as e:
            return {
                "tool": self.name,
                "input": word,
                "output": f"Error: {str(e)}"
            }
