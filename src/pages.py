import os
import time
import random
import asyncio
import kagglehub
import pandas as pd
import plotly.express as px
import shutil
from nicegui import ui, run
from src.app_state import app_state, Question, ModelTraining
from src.open_router import open_router
from src.model_train import train_model, infer_model, grad_cam, Models
from torchvision.datasets import ImageFolder

def model_page(): 
    async def on_train_model(model_training: ModelTraining): 
        model = model_training.get_model_function()
        (training_loss, training_accuracy, validation_loss, validation_accuracy, training_time) = await run.cpu_bound(train_model, app_state, model, model_training.model_file_name, model_training)
        
        model_training.training_loss = training_loss
        model_training.training_accuracy = training_accuracy
        model_training.validation_loss = validation_loss
        model_training.validation_accuracy = validation_accuracy
        model_training.training_time = training_time
        
        print(f"Training time: {training_time} seconds")
        print(f"validation accuracy: {validation_accuracy * 100}%")
        print(f"validation loss: {validation_loss}")
        app_state.save()
        
        ui.notify("Model trained", color="positive")
        
        
    def reduce_list_length(list: list[str], length: int):
        if len(list) <= length:
            return list
        else:
            return_list = []
            for i, val in enumerate(list):
                if i % 2 == 0:
                    return_list.append(val)
            return return_list
        
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
                
                loss_data = {}
                for model_training in [model for model in trained_models if len(model_training.training_loss) > 0]:
                    loss_data[model_training.model_name] = model_training.training_loss
                    loss_data[model_training.model_name] = reduce_list_length(loss_data[model_training.model_name], 10)
                loss_df = pd.DataFrame(loss_data)
                
                acc_data = {}
                for model_training in [model for model in trained_models if len(model_training.training_accuracy) > 0]:
                    acc_data[model_training.model_name] = [acc * 100 for acc in model_training.training_accuracy]
                    acc_data[model_training.model_name] = reduce_list_length(acc_data[model_training.model_name], 10)
                acc_df = pd.DataFrame(acc_data)
                
                val_loss_data = {}
                for model_training in [model for model in trained_models if len(model_training.validation_loss) > 0]:
                    val_loss_data[model_training.model_name] = model_training.validation_loss
                    val_loss_data[model_training.model_name] = reduce_list_length(val_loss_data[model_training.model_name], 10)
                val_loss_df = pd.DataFrame(val_loss_data)
                
                val_acc_data = {}
                for model_training in [model for model in trained_models if len(model_training.validation_accuracy) > 0]:
                    val_acc_data[model_training.model_name] = [acc * 100 for acc in model_training.validation_accuracy]
                    val_acc_data[model_training.model_name] = reduce_list_length(val_acc_data[model_training.model_name], 10)
                val_acc_df = pd.DataFrame(val_acc_data)
                
                time_data = {}
                for model_training in trained_models:
                    time_data[model_training.model_name] = model_training.training_time
                
                # Create accuracy chart
                fig_accuracy = px.bar(chart_model, x="model_name", 
                                                   y="accuracy", 
                                                   range_y=[0, 100],
                                                   color="accuracy",
                                                   color_continuous_scale=["red","yellow", "green"],
                                                   title="Model Accuracy",
                                                   labels={"accuracy": "Accuracy", "model_name": "Model"})
                ui.plotly(fig_accuracy)
                
                # Create loss chart
                fig_loss = px.bar(chart_model, x="model_name", 
                                               y="loss", 
                                               color="loss",
                                               color_continuous_scale=["green","yellow", "red"],
                                               title="Model Loss",
                                               labels={"loss": "Loss", "model_name": "Model"})
                ui.plotly(fig_loss)
                
                fig_training_loss = px.line(loss_df, y=loss_df.columns, 
                                            title="Model Training Loss",
                                            labels={"index": "Epoch", "value": "Loss", "variable": "Model"})
                fig_training_loss.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                
                ui.plotly(fig_training_loss)
                
                fig_training_accuracy = px.line(acc_df, y=acc_df.columns, 
                                                range_y=[0, 100],
                                                title="Model Training Accuracy",
                                                labels={"index": "Epoch", "value": "Accuracy", "variable": "Model"})
                fig_training_accuracy.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                ui.plotly(fig_training_accuracy)

                fig_validation_loss = px.line(val_loss_df, y=val_loss_df.columns, 
                                                title="Model Validation Loss",
                                                labels={"index": "Epoch", "value": "Loss", "variable": "Model"})
                fig_validation_loss.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                ui.plotly(fig_validation_loss)

                fig_validation_accuracy = px.line(val_acc_df, y=val_acc_df.columns, 
                                                range_y=[0, 100],
                                                title="Model Validation Accuracy",
                                                labels={"index": "Epoch", "value": "Accuracy", "variable": "Model"})
                fig_validation_accuracy.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                ui.plotly(fig_validation_accuracy)
                
                
                fig_training_time = px.bar(x=time_data.keys(), y=time_data.values(),  
                                            title="Training Time (seconds)",
                                            color=time_data.values(),
                                            color_continuous_scale=["green","yellow", "red"],
                                            labels={"x": "Model", "y": "Training Time", "color": "Training Time"})
                ui.plotly(fig_training_time)
        

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
        path = kagglehub.dataset_download("shawngano/gano-cat-breed-image-collection") 

        # Copy data to local folder using a relative path
        local_path = "./data"
        
        # Check if directory exists before removing it
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        
        # Copy the data (this will create the directory)
        shutil.copytree(path, local_path)  
        path = local_path 

        cat_pathname = os.path.join(path, os.listdir(path)[0])
        app_state.image_path = cat_pathname
        
        image_folder = ImageFolder(cat_pathname)
        
        # app_state.class_names =  [pathname for pathname in os.listdir(cat_pathname) if os.path.isdir(os.path.join(cat_pathname, pathname))]
        app_state.class_names = image_folder.classes
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
count_down_time = 15 
 
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
    #llm_answer = "1|This is a test answer"
    return llm_answer   

def get_model_answer(question: Question | None, model_name: str) -> str | None:
    if (question is None):
        return None
    model_answer = infer_model(app_state, model_name, question)
    if (model_answer is None):
        return None
    return model_answer


def question_page():
    global questions
    global selected_question
    global count_down_time
    
    # Add countdown timer state
    countdown_active = False
    countdown_timer = None
    countdown_label = None
    remaining_time = count_down_time
    
    result_dialog = ui.dialog().props("maximized persistent")
    
    def countdown_timer_callback():
        nonlocal remaining_time, countdown_active
        if not countdown_active or remaining_time <= 0:
            return
        
        if countdown_label:
            countdown_label.text = f"Time remaining: {remaining_time} seconds"
        
        remaining_time -= 1
        
        if remaining_time <= 0 and countdown_active:
            # Time ran out, submit a random answer
            if selected_question and selected_question.user_answer is None:
                ui.notify("Time's up! Submitting random answer.", color="warning")
                # Use a random answer and ensure the popup is shown
                random_answer = random.randint(0, 3)
                selected_question.user_answer = random_answer
                show_answer_dialog()
            # Make sure to stop the countdown when it completes
            stop_countdown()
    
    def start_countdown():
        nonlocal countdown_active, remaining_time, countdown_timer
        countdown_active = True
        remaining_time = count_down_time
        
        # Cancel existing timer if any
        if countdown_timer:
            countdown_timer.cancel()
        
        # Create a new timer that runs every second
        countdown_timer = ui.timer(1.0, countdown_timer_callback)
    
    def stop_countdown():
        nonlocal countdown_active, countdown_timer
        countdown_active = False
        if countdown_timer:
            countdown_timer.cancel()
            countdown_timer = None
    
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
        with ui.column().classes("w-full bg-grey").style("height: calc(100vh - 150px);"):
            with ui.image(selected_question.image_path).classes("rounded-lg").props("fit=contain"):
                nonlocal countdown_label
                countdown_label = ui.label(f"What is this cat breed?").classes("absolute-top text-subtitle1 text-center text-bold")
                # ui.label("What is this cat breed?").classes("absolute-top text-subtitle1 text-center")
                with ui.grid().classes("grid-cols-2 gap-4 w-full absolute bottom-0 left-0"):
                    for i, choice in enumerate(selected_question.choices):
                        ui.button(choice, on_click=lambda i=i: submit_answer(i)).classes("font-bold")
        
    async def run_llm_answer():
        start_time = time.time()
        llm_answer = await run.cpu_bound(get_llm_answer, selected_question) 
        llm_answer_label.text = llm_answer
        
        selected_question.llm_answer = int(llm_answer.split("|")[0])
        selected_question.llm_rational = llm_answer.split("|")[1]
        selected_question.llm_answer_time = time.time() - start_time
        
    async def run_model_answer():
        start_time = time.time()
        model_answer = await run.cpu_bound(get_model_answer, selected_question, Models.ClassificationModel)  
        selected_question.model_answer0 = 0 if model_answer.lower() == selected_question.choices[0].lower()  else \
                                          1 if model_answer.lower() == selected_question.choices[1].lower()  else \
                                          2 if model_answer.lower() == selected_question.choices[2].lower()  else \
                                          3 if model_answer.lower() == selected_question.choices[3].lower()  else -1
        selected_question.model_time0 = time.time() - start_time
        
        start_time = time.time()
        model_answer = await run.cpu_bound(get_model_answer, selected_question, Models.ResNetModel)  
        selected_question.model_answer1 = 0 if model_answer.lower() == selected_question.choices[0].lower()  else \
                                          1 if model_answer.lower() == selected_question.choices[1].lower()  else \
                                          2 if model_answer.lower() == selected_question.choices[2].lower()  else \
                                          3 if model_answer.lower() == selected_question.choices[3].lower()  else -1
        selected_question.model_time1 = time.time() - start_time
        
        start_time = time.time()
        model_answer = await run.cpu_bound(get_model_answer, selected_question, Models.EfficientNetModel)  
        selected_question.model_answer2 = 0 if model_answer.lower() == selected_question.choices[0].lower()  else \
                                          1 if model_answer.lower() == selected_question.choices[1].lower()  else \
                                          2 if model_answer.lower() == selected_question.choices[2].lower()  else \
                                          3 if model_answer.lower() == selected_question.choices[3].lower()  else -1
        selected_question.model_time2 = time.time() - start_time
        
        start_time = time.time()
        model_answer = await run.cpu_bound(get_model_answer, selected_question, Models.VGGModel)  
        selected_question.model_answer3 = 0 if model_answer.lower() == selected_question.choices[0].lower()  else \
                                          1 if model_answer.lower() == selected_question.choices[1].lower()  else \
                                          2 if model_answer.lower() == selected_question.choices[2].lower()  else \
                                          3 if model_answer.lower() == selected_question.choices[3].lower()  else -1
        selected_question.model_time3 = time.time() - start_time

    def answer_panel(answer, title: str, rational: str | None = None, time_taken: float | None = None, grad_cam_model: str | None = None):
        print(f"answer: {answer}, title: {title} {"({time_taken:.2f} seconds)" if time_taken is not None else ""}, rational: {rational}, time_taken: {time_taken}, grad_cam_model: {grad_cam_model}, correct_answer: {selected_question.correct_answer}")
        # with ui.card().classes("w-full mb-4 p-4"):
        ui.label(f"{title} {f'({time_taken:.2f} seconds)' if time_taken is not None else ''}").classes("text-h6 text-primary")
        with ui.row().classes("items-center"):
            ui.label(f"{selected_question.choices[answer]}" if answer != -1 else "I don't know").classes("text-h5 font-bold")
            if (answer == selected_question.correct_answer):
                ui.icon("check_circle").classes("text-positive text-h5 ml-2")
                ui.label("Correct!").classes("text-positive text-h6 ml-2")
            else:
                ui.icon("cancel").classes("text-negative text-h5 ml-2")
                ui.label("Incorrect!").classes("text-negative text-h6 ml-2")
                ui.label(f"The correct answer was: {selected_question.choices[selected_question.correct_answer]}").classes("text-body1 mt-2 text-weight-medium")
            
        if (rational is not None):
            ui.label(rational).classes("text-body1 text-weight-medium")
             
        if (grad_cam_model is not None):
            paint_grad_cam(grad_cam_model)
            
    def show_answer_dialog():
        # Create a dialog to show results
        with result_dialog, ui.card():
            with ui.card_section().classes("scroll"):
                with ui.grid().classes("w-full grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2"):
                
                    with ui.card().classes("w-full mb-4 p-4"):
                        answer_panel(selected_question.user_answer, "Your Answer") 
                    
                    # Create placeholder for LLM answer
                    with ui.card().classes("w-full mb-4 p-4") as llm_card:
                        ui.label("ChatGPT-4o-mini Answer").classes("text-h6 text-primary mb-2")
                        ui.label("Loading...").classes("text-h5 font-bold")
                    
                    # Create placeholders for model answers that will be updated after inference
                    model_answer_cards = []
                    for i, model_name in enumerate(["Classification Model", "ResNet Model", "EfficientNet Model", "VGG Model"]):
                        with ui.card().classes("w-full mb-4 p-4") as card:
                            ui.label(f"{model_name} Answer").classes("text-h6 text-primary mb-2")
                            ui.label("Loading...").classes("text-h5 font-bold")
                            model_answer_cards.append(card)
                    
                    # Run LLM and model inference and update UI when complete
                    async def run_inference_and_update():
                        # Run LLM inference
                        await run_llm_answer()
                        # Update LLM card
                        llm_card.clear()
                        with llm_card:
                            answer_panel(selected_question.llm_answer, "ChatGPT-4o-mini Answer", 
                                        rational=selected_question.llm_rational, time_taken=selected_question.llm_answer_time)
                        
                        # Run model inference
                        await run_model_answer()
                        # Update the model answer cards with results
                        model_results = [
                            (selected_question.model_answer0, selected_question.model_time0, Models.ClassificationModel),
                            (selected_question.model_answer1, selected_question.model_time1, Models.ResNetModel),
                            (selected_question.model_answer2, selected_question.model_time2, Models.EfficientNetModel),
                            (selected_question.model_answer3, selected_question.model_time3, Models.VGGModel)
                        ]
                        
                        for i, (model_answer, model_time, grad_cam_model) in enumerate(model_results):
                            # Clear the card content
                            model_answer_cards[i].clear()
                            # Add the updated content
                            with model_answer_cards[i]:
                                answer_panel(model_answer, f"{['Classification', 'ResNet', 'EfficientNet', 'VGG'][i]} Model Answer", 
                                            time_taken=model_time, grad_cam_model=grad_cam_model)
                        
                        # Refresh the result label to show updated statistics
                        paint_result_label.refresh()
                        
                    # Start the inference process
                    asyncio.create_task(run_inference_and_update())
                
            with ui.card_actions().classes("justify-between w-full gap-2 mt-4"):
                paint_result_label(result_dialog)
            result_dialog.open()
          
    @ui.refreshable
    def paint_result_label(result_dialog):
        global questions 
        user_correct = sum(1 for question in questions if question.user_answer == question.correct_answer)
        llm_correct = sum(1 for question in questions if question.llm_answer == question.correct_answer)
        model1_correct = sum(1 for question in questions if question.model_answer0 == question.correct_answer)
        model2_correct = sum(1 for question in questions if question.model_answer1 == question.correct_answer)
        model3_correct = sum(1 for question in questions if question.model_answer2 == question.correct_answer)
        model4_correct = sum(1 for question in questions if question.model_answer3 == question.correct_answer)
        
        # Debug print removed for cleaner code
        with ui.row().classes("justify-between w-full"):
            with ui.row().classes("justify-start"):
                ui.label("You :").classes("text-h6 text-primary font-medium font-bold")
                ui.label(f"{user_correct}").classes("text-h6 text-primary font-medium text-black")
                ui.label("ChatGPT :").classes("text-h6 text-primary font-medium font-bold")
                ui.label(f"{llm_correct}").classes("text-h6 text-primary font-medium text-black")
                ui.label("Classification :").classes("text-h6 text-primary font-medium font-bold")
                ui.label(f"{model1_correct}").classes("text-h6 text-primary font-medium text-black")
                ui.label("ResNet :").classes("text-h6 text-primary font-medium font-bold")
                ui.label(f"{model2_correct}").classes("text-h6 text-primary font-medium text-black")
                ui.label("EfficientNet :").classes("text-h6 text-primary font-medium font-bold")
                ui.label(f"{model3_correct}").classes("text-h6 text-primary font-medium text-black")
                ui.label("VGG :").classes("text-h6 text-primary font-medium font-bold")
                ui.label(f"{model4_correct}").classes("text-h6 text-primary font-medium text-black")
                 
            ui.button("Next Question", on_click=lambda: next_question(result_dialog)).props("color=primary icon=navigate_next")
        
        # ui.label(f"Score You: {user_correct} | ChatGPT: {llm_correct} | Classification: {model1_correct} | ResNet: {model2_correct} | EfficientNet: {model3_correct} | VGG: {model4_correct}").classes("text-h6 text-primary font-medium")
        
    def submit_answer(answer):
        global selected_question
        # Stop the countdown when an answer is submitted
        stop_countdown()
        
        selected_question.user_answer = answer
        show_answer_dialog()
    
    def next_question(result_dialog):
        global questions
        global selected_question
        questions.append(random_question())
        selected_question = questions[-1]
        
        result_dialog.clear()
        result_dialog.close()
         
        paint_question_panel.refresh()
        paint_grad_cam.refresh()
        # Remove LLM inference from here - it will be done when answer is submitted
        # asyncio.create_task(run_llm_answer())
        # Start the countdown for the new question
        start_countdown()
        
    questions.append(random_question())
    selected_question = questions[0]
    paint_question_panel()  
    # Remove LLM inference from here - it will be done when answer is submitted
    # asyncio.create_task(run_llm_answer())
    # Start the countdown for the first question
    start_countdown()
    
    @ui.refreshable
    def paint_grad_cam(model_name: str):
        (image, _) = grad_cam(app_state, model_name, selected_question)
        ui.image(image).classes("rounded-lg h-[calc(20vh)]").props("fit=contain")
    
    llm_answer_label = ui.label("Loading LLM Answer...").classes("text-h6 hidden")

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
    
