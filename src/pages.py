from nicegui import ui
import kagglehub
import os
import random
from src.app_state import app_state, Question
from src.open_router import open_router

def model_page():
    with ui.grid().classes("gap-2 md:gap-4 grid-cols-2 md:grid-cols-4"):
        ui.button("Load Model", on_click=lambda: ui.navigate.back())
        ui.button("Train New Model", on_click=lambda: ui.navigate.back())
        ui.button("Load Data", on_click=lambda: ui.navigate.back())

    log_text = ui.textarea(
        label="Log",
        placeholder="Log", 
    ).classes("w-full")
    
 
def kaggle_page():      
    def update_class_select(e): 
        app_state.kaggle_selected_class = e.value
        ui.notify(app_state.kaggle_selected_class)
        paint_image.refresh()
    
    @ui.refreshable
    def paint_image():  
        if (app_state.kaggle_selected_class is not None):
            with ui.card().classes("w-full"):
                ui.label(app_state.kaggle_selected_class).classes("text-h6") 
                
                selected_path = os.path.join(app_state.image_path, app_state.kaggle_selected_class)  
                image_files = os.listdir(selected_path)
                
                with ui.grid().classes("gap-2 md:gap-4 w-full grid-cols-2 md:grid-cols-4") as image_grid:
                    for _, image_file in enumerate(image_files[:20]):
                        image_path = os.path.join(selected_path, image_file)
                        ui.image(image_path).style("width: 100%; height: 100%;").props("ratio=1")
                    
        else:
            ui.label("Please select a class")
    
    def load_data():
        print("Loading data...")
        path = kagglehub.dataset_download("shawngano/gano-cat-breed-image-collection") 
        cat_pathname = os.path.join(path, os.listdir(path)[0])
        app_state.image_path = cat_pathname
        app_state.class_names = [pathname for pathname in os.listdir(cat_pathname) if os.path.isdir(os.path.join(cat_pathname, pathname))]
        app_state.save()
        ui.notify("Data loaded", color="positive")

    with ui.card().classes("w-full"):
        with ui.row().classes("justify-between w-full"):
            ui.label("Kaggle").classes("text-h6")
            ui.button("Download from Kaggle", icon="download", on_click=load_data)
        ui.select(app_state.class_names,label="Pick a Class", value=app_state.kaggle_selected_class, on_change=update_class_select).classes("w-full")
        
    paint_image()
    
questions : list[Question] = []

def question_page():
    global questions
    
    def random_question() -> Question:
        random_class = random.choice(app_state.class_names) 
        image_path = os.path.join(app_state.image_path, random_class, random.choice(os.listdir(os.path.join(app_state.image_path, random_class))))

        class_names = app_state.class_names.copy()
        class_names.remove(random_class)
        
        random_choices = random.sample(class_names, 3)
        random_choices.append(random_class)
        random.shuffle(random_choices)
        
        correct_answer = random_choices.index(random_class)
        
        return Question(image_path, random_choices[0], random_choices[1], random_choices[2], random_choices[3], correct_answer)
    
    questions.append(random_question())
    
    with ui.card().classes("w-full"):
        with ui.column().classes("w-full"):
            ui.label("Question Page").classes("text-h6")
            ui.button("Question Page", on_click=lambda: ui.navigate.back())
            
            ui.label("Question").classes("text-h6")
            ui.label(questions[0].image_path).classes("text-h6")
            
        

def open_router_page():

    @ui.refreshable
    def open_router_panel():
        
        def get_random_cat_image():
            random_class = random.choice(app_state.class_names) 
            image_path = os.path.join(app_state.image_path, random_class, random.choice(os.listdir(os.path.join(app_state.image_path, random_class))))
            return (random_class, image_path) 
        
        def open_router_response(query_text, image_path):
            open_router_response = open_router.get_response(query_text, image_path)
            ui.notify(open_router_response)
            response_text.value = open_router_response
        
        def refresh_image():
            open_router_panel.refresh()
            
        with ui.card().classes("w-full"):
            (random_class, image_path) = get_random_cat_image()
            if (image_path is not None):
                with ui.card().classes("relative w-full h-96 p-0 m-0"):
                    ui.image(image_path).classes("h-96 rounded-lg")
                    ui.label(random_class).classes("text-h6 absolute bottom-0 left-0 bg-black bg-opacity-50 text-white w-full p-2 rounded-b-lg")
            
            ui.label("Open Router").classes("text-h6")
            with ui.grid().classes("grid-cols-2 gap-4 w-full"):
                query_text = ui.textarea(value="What is this cat breed?", label="Message", placeholder="Message").classes("w-full").props("outlined")
                response_text = ui.textarea(label="Response", placeholder="Response").classes("w-full").props("outlined")
            with ui.row().classes("justify-end w-full"):
                ui.button("Reset", icon="refresh", on_click=refresh_image)
                ui.button("Send", icon="send", on_click=lambda: open_router_response(query_text.value, image_path))
                


        
    open_router_panel()
    
    
