from qtconsole import inprocess


class JupyterConsoleWidget(inprocess.QtInProcessRichJupyterWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.kernel_manager = inprocess.QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client = self.kernel_manager.client()
        self.kernel_client.start_channels()
        # force dark mode, with foreground and background
        # modified to be easier on eyes (values set from KDE dark breeze)
        self.set_default_style("linux")
        css = self.styleSheet()
        new_css = css.replace("black", "/#232627").replace("white", "/#eff0f1")
        self.setStyleSheet(new_css)

    def shutdown_kernel(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()
