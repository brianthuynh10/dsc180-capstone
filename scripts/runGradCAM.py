from interpretability.GradCAM.layer_progression import compare_layers_progression
from interpretability.GradCAM.group_comparison import compare_groups

def main(): 
    compare_layers_progression()
    compare_groups()
    
if __name__ == "__main__":
    main()