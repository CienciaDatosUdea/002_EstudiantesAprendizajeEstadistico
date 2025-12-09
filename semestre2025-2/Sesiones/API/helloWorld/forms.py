
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, validators



class AddTaskForm(FlaskForm):

    title  = StringField('Title', validators = [validators.DataRequired()]   )
    submit = SubmitField('Submit')


