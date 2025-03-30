from nicegui import ui, run
import kagglehub
import os
import random
import asyncio
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
    def update_class_select(class_name): 
        app_state.kaggle_selected_class = class_name
        ui.notify(app_state.kaggle_selected_class)
        paint_image.refresh()
        paint_class_select.refresh()
    
    @ui.refreshable
    def paint_image():  
        if (app_state.kaggle_selected_class is not None):
            with ui.card().classes("w-full"): 
                selected_path = os.path.join(app_state.image_path, app_state.kaggle_selected_class)  
                image_files = os.listdir(selected_path)
                
                with ui.grid().classes("gap-2 md:gap-4 w-full grid-cols-2 md:grid-cols-4") as image_grid:
                    for _, image_file in enumerate(image_files[:20]):
                        image_path = os.path.join(selected_path, image_file)
                        ui.image(image_path).props("ratio=1").classes("rounded-lg")
                    
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
        
    @ui.refreshable
    def paint_class_select():
        # ui.select(app_state.class_names,label="Pick a Class", value=app_state.kaggle_selected_class, on_change=update_class_select).classes("w-full")
        with ui.row().classes("gap-2 w-full"):
            for class_name in app_state.class_names:
                ui.button(class_name, on_click=lambda class_name=class_name: update_class_select(class_name)).props("outline size=sm" if class_name != app_state.kaggle_selected_class else "filled size=sm")
        
    with ui.card().classes("w-full"):
        with ui.row().classes("justify-between w-full"):
            ui.label("Kaggle").classes("text-h6")
            ui.button("Download from Kaggle", icon="download", on_click=load_data)
        
        paint_class_select()
        
    paint_image()
    
questions : list[Question] = []
selected_question : Question = None

def get_llm_answer(question: Question | None) -> str | None:
    if (question is None):
        return None
    question_text = f'''What is this cat breed?
    0:{question.choices[0]}
    1:{question.choices[1]}
    2:{question.choices[2]}
    3:{question.choices[3]}
    
    ANSWER IN THIS FORMAT:
    
    choice|rational
    
    where choice is 0, 1, 2, or 3 and rational is a short rational for the answer'''
    llm_answer = open_router.get_response(question_text, question.image_path)
    print(llm_answer)
    return llm_answer   

def question_page():
    global questions
    global selected_question
    def random_question() -> Question:
        random_class = random.choice(app_state.class_names) 
        image_path = os.path.join(app_state.image_path, random_class, random.choice(os.listdir(os.path.join(app_state.image_path, random_class))))

        class_names = app_state.class_names.copy()
        class_names.remove(random_class)
        
        random_choices = random.sample(class_names, 3)
        random_choices.append(random_class)
        random.shuffle(random_choices)
        
        correct_answer = random_choices.index(random_class)
        
        return Question(image_path, random_choices, correct_answer)
    
    @ui.refreshable
    def paint_question_panel():
        global selected_question
        with ui.column().classes("w-full").style("height: calc(100vh - 150px);"):
            with ui.image(selected_question.image_path).classes("rounded-lg").props("ratio=1"):
                ui.label("What is this cat breed?").classes("absolute-top text-subtitle1 text-center")
                with ui.grid().classes("grid-cols-2 gap-4 w-full absolute bottom-0 left-0"):
                    for i, choice in enumerate(selected_question.choices):
                        ui.button(choice, on_click=lambda i=i: submit_answer(i)).props("outline")
        
    async def run_llm_answer():
        llm_answer = await run.cpu_bound(get_llm_answer, selected_question) 
        llm_answer_label.text = llm_answer
                    
    def submit_answer(answer):
        global selected_question
        selected_question.user_answer = answer
        with ui.dialog() as dialog, ui.card():
            with ui.column().classes("w-full"):
                ui.label(f"You answered: {selected_question.choices[answer]}")
                if (answer == selected_question.correct_answer):
                    ui.label("Correct!")
                else:
                    ui.label("Incorrect!")
                    
                ui.button("Next", on_click=next_question)
                ui.button("Close", on_click=lambda: dialog.close())
            dialog.open()
    
    def next_question():
        global questions
        global selected_question
        questions.append(random_question())
        selected_question = questions[-1]
        paint_question_panel.refresh()
        
    questions.append(random_question())
    selected_question = questions[0]
    paint_question_panel() 
    llm_answer_label = ui.label() 
            
        

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
    
    
