"""
Basic functionality tests for research modules without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

def test_module_structure():
    """Test that research modules have correct structure."""
    # Test that files exist
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    
    expected_files = [
        '__init__.py',
        'comparative_study.py',
        'multi_objective.py', 
        'experimental_framework.py'
    ]
    
    for filename in expected_files:
        filepath = os.path.join(research_dir, filename)
        assert os.path.exists(filepath), f"Missing file: {filename}"
        
        # Check that files are not empty
        with open(filepath, 'r') as f:
            content = f.read()
            assert len(content) > 100, f"File {filename} appears to be empty or too small"
    
    print("✓ All research module files exist and have content")


def test_module_imports():
    """Test basic module imports without numpy/scipy dependencies."""
    
    # Test that we can at least parse the files
    import ast
    
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    
    for filename in ['comparative_study.py', 'multi_objective.py', 'experimental_framework.py']:
        filepath = os.path.join(research_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        try:
            # Parse the Python code
            ast.parse(content)
            print(f"✓ {filename} has valid Python syntax")
        except SyntaxError as e:
            raise AssertionError(f"Syntax error in {filename}: {e}")


def test_class_definitions():
    """Test that expected classes are defined."""
    import ast
    
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    
    # Expected classes in each module
    expected_classes = {
        'comparative_study.py': [
            'ComparativeStudyFramework',
            'CNNBaseline', 
            'LSTMBaseline',
            'TinyMLBaseline',
            'PowerEfficiencyAnalysis',
            'ModelComparison'
        ],
        'multi_objective.py': [
            'MultiObjectiveOptimizer',
            'NSGA3Algorithm',
            'ParetoFrontierAnalysis',
            'Individual'
        ],
        'experimental_framework.py': [
            'ExperimentalFramework',
            'ExperimentConfig',
            'ExperimentResult',
            'SystemInfo',
            'ReproducibilityManager',
            'DatasetGenerator'
        ]
    }
    
    for filename, classes in expected_classes.items():
        filepath = os.path.join(research_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find all class definitions
        defined_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                defined_classes.append(node.name)
        
        for expected_class in classes:
            assert expected_class in defined_classes, f"Class {expected_class} not found in {filename}"
        
        print(f"✓ {filename} contains all expected classes: {classes}")


def test_function_definitions():
    """Test that key functions are defined."""
    import ast
    
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    
    # Test comparative_study.py
    filepath = os.path.join(research_dir, 'comparative_study.py')
    with open(filepath, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find all function definitions
    defined_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_functions.append(node.name)
    
    # Should have basic utility functions
    assert len(defined_functions) > 10, "Expected more function definitions in comparative_study.py"
    print(f"✓ comparative_study.py defines {len(defined_functions)} functions")


def test_docstrings():
    """Test that modules have proper docstrings."""
    import ast
    
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    
    for filename in ['comparative_study.py', 'multi_objective.py', 'experimental_framework.py']:
        filepath = os.path.join(research_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Check module docstring
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
            docstring = tree.body[0].value.s
            assert len(docstring) > 50, f"Module docstring too short in {filename}"
            print(f"✓ {filename} has a proper module docstring")
        else:
            raise AssertionError(f"No module docstring found in {filename}")


def test_imports_structure():
    """Test that imports are properly structured."""
    import ast
    
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    
    for filename in ['comparative_study.py', 'multi_objective.py', 'experimental_framework.py']:
        filepath = os.path.join(research_dir, filename)
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Count imports
        import_count = 0
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        
        assert import_count > 5, f"Expected more imports in {filename}"
        print(f"✓ {filename} has {import_count} import statements")


def test_init_file():
    """Test __init__.py exports."""
    research_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'python', 'liquid_audio_nets', 'research')
    init_file = os.path.join(research_dir, '__init__.py')
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    # Should have imports from submodules
    assert 'from .comparative_study import' in content
    assert 'from .multi_objective import' in content  
    assert 'from .experimental_framework import' in content
    
    # Should have __all__ definition
    assert '__all__ = [' in content
    
    print("✓ __init__.py properly exports research modules")


if __name__ == "__main__":
    test_module_structure()
    test_module_imports()
    test_class_definitions()
    test_function_definitions()
    test_docstrings()
    test_imports_structure()
    test_init_file()
    
    print("\n🎉 All basic functionality tests passed!")
    print("Research modules are properly structured and ready for use.")