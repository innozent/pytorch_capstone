import os
from nicegui import ui, app
import src.pages as pages
import torch
from src.app_state import app_state

@ui.page("/")
def index_page():
    set_html_asset()
    main_container()
    with ui.header():
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
        ui.label("Cat Breed Quiz").classes("text-h5 font-bold playwrite-hu")
        
    with ui.left_drawer(fixed=True).classes("border-r border-gray-200") as left_drawer:
        paint_navigation_drawer()

@ui.refreshable
def paint_navigation_drawer():
    with ui.list().classes("w-full"):
        ui.item("Open Router",on_click=lambda: set_page("open_router")).classes("font-bold text-lg text-accent" if app_state.current_page == "open_router" else "")
        ui.item("Kaggle",on_click=lambda: set_page("kaggle")).classes("font-bold text-lg text-accent" if app_state.current_page == "kaggle" else "")
        ui.item("Model",on_click=lambda: set_page("model")).classes("font-bold text-lg text-accent" if app_state.current_page == "model" else "")
        ui.item("Question",on_click=lambda: set_page("question")).classes("font-bold text-lg text-accent" if app_state.current_page == "question" else "")

@ui.refreshable
def main_container():
    with ui.column().classes("w-full p-4"): 
        if (app_state.current_page is None):
            ui.label("Welcome to the Model Interface").classes("text-h4 q-mb-md")
            ui.label("Use the navigation drawer on the left to access different sections of the application.")
        elif (app_state.current_page == "model"):
            pages.model_page()
        elif (app_state.current_page == "question"):
            pages.question_page()
        elif (app_state.current_page == "kaggle"):
            pages.kaggle_page()
        elif (app_state.current_page == "open_router"):
            pages.open_router_page()
def set_page(page):
    app_state.current_page = page
    main_container.refresh()
    paint_navigation_drawer.refresh()
    ui.notify("Page changed to " + page)

def set_html_asset():
    ui.add_head_html(''' 
                    <link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,100..900;1,100..900&family=Playwrite+HU:wght@100..400&display=swap" rel="stylesheet">''')
    ui.add_css(''' 
               .playwrite-hu {
                    font-family: 'Playwrite HU', serif;
               }
               
               body {
                    font-family: 'Noto Sans', sans-serif;
               }
               ''')
    ui.colors(primary="darkorange")
    
if __name__ == "__main__" or __name__ == "__mp_main__":
    app_state.load() 
    app_state.current_page = "kaggle"
    # app_state.current_page = "question"
    ui.run(title="Cat Breed Quiz")  
    
