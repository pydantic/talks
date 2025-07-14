from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import (
    Header,
    Footer,
    Input,
    TextArea,
    Label,
    Button,
    Static,
    ListView,
    ListItem,
)


class BlogPostForm(App):
    CSS_PATH = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with VerticalScroll(id="form"):
            yield Label("Blog Post Topic / Headline")
            self.topic_input = Input(placeholder="Enter the blog post topic...", id="topic")
            yield self.topic_input

            yield Label("Author Name")
            self.author_name_input = Input(placeholder="Enter author name...", id="author_name")
            yield self.author_name_input

            yield Label("Author Role")
            self.author_role_input = Input(placeholder="Enter author role...", id="author_role")
            yield self.author_role_input

            yield Label("Content Guidance")
            self.content_guidance_input = Input(placeholder="Enter content guidance...", id="content_guidance")
            yield self.content_guidance_input

            yield Label("Additional Requirements / Direction")
            self.additional_req_input = Input(placeholder="Enter additional requirements...", id="additional_requirements")
            yield self.additional_req_input

            # yield Label("Reference Links")
            # self.links_container = Container(id="links_container")
            # self.link_inputs = []
            # self.add_link_input()
            # yield self.links_container
            # yield Button(label="Add Link", id="add_link")
            
            yield Button(label="Submit", id="submit_btn", variant="success")
            self.output_area = Static(id="output")
            yield self.output_area

    def add_link_input(self):
        input_field = Input(placeholder="Enter a reference link...", id=f"link_{len(self.link_inputs)}")
        self.link_inputs.append(input_field)
        self.links_container.mount(input_field)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add_link":
            self.add_link_input()
        elif event.button.id == "submit_btn":
            await self.display_output()

    async def display_output(self):
        topic = self.topic_input.value
        author_name = self.author_name_input.value
        author_role = self.author_role_input.value
        content_guidance = self.content_guidance_input.value
        additional_req = self.additional_req_input.value
        links = [link.value for link in self.link_inputs if link.value.strip() != ""]

        result = (
            f"# Blog Post Submission\n"
            f"**Topic:** {topic}\n"
            f"**Author:** {author_name} ({author_role})\n\n"
            f"## Content Guidance\n{content_guidance}\n\n"
            f"## Additional Requirements\n{additional_req}\n\n"
            f"## Reference Links:\n"
        )
        for link in links:
            result += f"- {link}\n"

        self.output_area.update(result)


if __name__ == "__main__":
    app = BlogPostForm()
    app.run()