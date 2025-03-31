from nicegui import ui, run
import kagglehub
import os
import random
import asyncio
from src.app_state import app_state, Question, ModelTraining
from src.open_router import open_router
from src.model_train import train_model, infer_model
import plotly.express as px
import time
import pandas as pd

def model_page(): 
    async def on_train_model(model_training: ModelTraining): 
        model = model_training.get_model_function()
        print(model_training.model_name)
        (training_loss, training_accuracy, validation_loss, validation_accuracy, training_time) = await run.cpu_bound(train_model, app_state, model, model_training.model_file_name)
        
        model_training.training_loss = training_loss
        model_training.training_accuracy = training_accuracy
        model_training.validation_loss = validation_loss
        model_training.validation_accuracy = validation_accuracy
        model_training.training_time = training_time
        app_state.save()
        
        ui.notify("Model trained", color="positive")
        
    def paint_accuracy_chart():
        
        trained_models = [model for model in app_state.model_trainings if model.training_loss is not None]
        if len(trained_models) == 0:
            ui.label("No model trained")
            return
        
        with ui.card().classes("w-full"):
            with ui.grid().classes("grid-cols-1 lg:grid-cols-2 gap-0 w-full"):
                for model_training in app_state.model_trainings:
                    model_training.accuracy = model_training.get_accuracy()
                    model_training.loss = model_training.get_loss()
                
                chart_model = {}
                chart_model["model_name"] = [model_training.model_name for model_training in trained_models]
                chart_model["accuracy"] = [model_training.accuracy * 100 for model_training in trained_models]
                chart_model["loss"] = [model_training.loss for model_training in trained_models]
                
                line_data = {}
                for model_training in trained_models:
                    line_data[model_training.model_name] = model_training.training_loss
                line_df = pd.DataFrame(line_data)
                
                acc_data = {}
                for model_training in trained_models:
                    acc_data[model_training.model_name] = [acc * 100 for acc in model_training.training_accuracy]
                acc_df = pd.DataFrame(acc_data)
                
                # Create accuracy chart
                fig_accuracy = px.bar(chart_model, x="model_name", 
                                                   y="accuracy", 
                                                   range_y=[0, 100],
                                                   color="accuracy",
                                                   title="Model Accuracy")
                ui.plotly(fig_accuracy)
                
                # Create loss chart
                fig_loss = px.bar(chart_model, x="model_name", 
                                               y="loss", 
                                               color="loss",
                                               title="Model Loss")
                ui.plotly(fig_loss)
                
                fig_training_loss = px.line(line_df, y=line_df.columns[1:], 
                                            title="Model Training Loss")
                
                ui.plotly(fig_training_loss)
                
                fig_training_accuracy = px.line(acc_df, y=acc_df.columns[1:], 
                                                range_y=[0, 100],
                                            title="Model Training Accuracy")
                ui.plotly(fig_training_accuracy)
                
        

    with ui.grid().classes("gap-2 md:gap-4 grid-cols-2 md:grid-cols-4"): 
        for model_training in app_state.model_trainings:
            ui.button(model_training.model_name, on_click=lambda model_training=model_training: on_train_model(model_training))
         
    paint_accuracy_chart()
    
 
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

def get_model_answer(question: Question | None) -> str | None:
    if (question is None):
        return None
    model_answer = infer_model(app_state, app_state.model_trainings[-1].model_name, question)
    if (model_answer is None):
        return None
    return model_answer


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
                        ui.button(choice, on_click=lambda i=i: submit_answer(i)).classes("font-bold")
        
    async def run_llm_answer():
        start_time = time.time()
        llm_answer = await run.cpu_bound(get_llm_answer, selected_question) 
        # llm_answer = "1|This is a test answer"
        # llm_answer_label.text = llm_answer
        
        selected_question.llm_answer = int(llm_answer.split("|")[0])
        selected_question.llm_rational = llm_answer.split("|")[1]
        selected_question.llm_answer_time = time.time() - start_time
        
    async def run_model_answer():
        start_time = time.time()
        model_answer = await run.cpu_bound(get_model_answer, selected_question)  
        selected_question.model_answer = 0 if model_answer.lower() == selected_question.choices[0].lower()  else \
                                         1 if model_answer.lower() == selected_question.choices[1].lower()  else \
                                         2 if model_answer.lower() == selected_question.choices[2].lower()  else \
                                         3 if model_answer.lower() == selected_question.choices[3].lower()  else -1
        selected_question.model_answer_time = time.time() - start_time
                 
    def answer_panel(answer, title: str):
        with ui.card().classes("w-full mb-4 p-4"):
            ui.label(title).classes("text-h6 text-primary mb-2")
            with ui.row().classes("items-center"):
                ui.label(f"{selected_question.choices[answer]}").classes("text-h5 font-bold")
                if (answer == selected_question.correct_answer):
                    ui.icon("check_circle").classes("text-positive text-h5 ml-2")
                    ui.label("Correct!").classes("text-positive text-h6 ml-2")
                else:
                    ui.icon("cancel").classes("text-negative text-h5 ml-2")
                    ui.label("Incorrect!").classes("text-negative text-h6 ml-2")
                    ui.label(f"The correct answer was: {selected_question.choices[selected_question.correct_answer]}").classes("text-body1 mt-2 text-weight-medium")
                
                 
    def submit_answer(answer):
        global selected_question
        selected_question.user_answer = answer
        with ui.dialog() as dialog, ui.card():
            with ui.column().classes("w-full p-4"):
                answer_panel(selected_question.user_answer, "Your Answer") 
                answer_panel(selected_question.llm_answer, "LLM Answer")
                answer_panel(selected_question.model_answer, "Model Answer")
                
                with ui.row().classes("justify-end w-full gap-2 mt-4"):
                    ui.button("Next Question", on_click=next_question).props("color=primary icon=navigate_next")
                    ui.button("Close", on_click=lambda: dialog.close()).props("flat icon=close")
            dialog.open()
    
    def next_question():
        global questions
        global selected_question
        questions.append(random_question())
        selected_question = questions[-1]
        paint_question_panel.refresh()
        asyncio.create_task(run_llm_answer())
        asyncio.create_task(run_model_answer())
        
    questions.append(random_question())
    selected_question = questions[0]
    paint_question_panel()  
    asyncio.create_task(run_llm_answer())
    asyncio.create_task(run_model_answer()) 

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
    
    
