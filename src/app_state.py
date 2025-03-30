import torch
import pickle

class Question:
    def __init__(self, image_path: str, choices: list[str], correct_answer: int):
        self.image_path : str = image_path
        self.choices : list[str] = choices
        self.correct_answer : int = correct_answer
        self.user_answer : int = None 
        self.llm_answer : int = None
        self.model_answer : int = None
        self.user_time : float = None
        self.llm_time : float = None
        self.model_time : float = None 
        self.llm_rational : str = None
    
    def is_user_correct(self):
        return self.user_answer == self.correct_answer
    
    def is_llm_correct(self):
        return self.llm_answer == self.correct_answer
    
    def is_model_correct(self):
        return self.model_answer == self.correct_answer  

class AppState:
    def __init__(self): 
        self.current_page = None
        self.device = torch.device("mps" if torch.mps.is_available() else 
                                   "cuda" if torch.cuda.is_available() else 
                                   "cpu")
        self.class_names = []
        self.image_path = "" 
        self.kaggle_selected_class = None
        
    def save(self): 
        with open("app_state.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self):
        with open("app_state.pkl", "rb") as f:
            loaded_state = pickle.load(f) 
            self.__dict__.update(loaded_state.__dict__)

app_state = AppState()

