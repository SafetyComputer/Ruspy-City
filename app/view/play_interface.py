from PySide6.QtCore import QSize
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget
from qfluentwidgets import PushButton, FluentIcon, TabBar, InfoBar, RoundMenu, Action, \
    TransparentDropDownToolButton

from app.common import MyFluentIcon
from app.widgets import *


class PlayInterface(QWidget):
    def __init__(self, text, parent):
        super().__init__(parent)
        self.setMinimumSize(750, 500)
        self.setMouseTracking(True)
        self.setObjectName(text.replace(' ', '-'))

        self.tab_bar = TabBar(self)
        self.board_interface = QStackedWidget(self, objectName='boardInterface')

        self.initShortcuts()
        self.initWidgets()
        self.initTab()

    def initWidgets(self):
        self.main_layout = QVBoxLayout()
        self.body_layout = QHBoxLayout()
        self.board_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        self.load_button = PushButton(self)
        self.load_button.setText("Import")
        self.load_button.setIcon(FluentIcon.DOWNLOAD)
        self.button_layout.addWidget(self.load_button)

        self.first_button = PushButton(self)
        self.first_button.setText('First')
        self.first_button.setIcon(MyFluentIcon.FIRST)
        self.button_layout.addWidget(self.first_button)

        self.previous_button = PushButton(self)
        self.previous_button.setText('Prev')
        self.previous_button.setIcon(MyFluentIcon.PREVIOUS)
        self.button_layout.addWidget(self.previous_button)

        self.next_button = PushButton(self)
        self.next_button.setText('Next')
        self.next_button.setIcon(MyFluentIcon.NEXT)
        self.button_layout.addWidget(self.next_button)

        self.last_button = PushButton(self)
        self.last_button.setText('Last')
        self.last_button.setIcon(MyFluentIcon.LAST)
        self.button_layout.addWidget(self.last_button)

        self.save_button = PushButton(self)
        self.save_button.setText("Save")
        self.save_button.setIcon(FluentIcon.SAVE)
        self.button_layout.addWidget(self.save_button)

        self.step_card = StepCard(self.onStepClicked)

        self.main_layout.addWidget(self.tab_bar)
        self.main_layout.addLayout(self.body_layout)

        self.body_layout.addLayout(self.board_layout)
        self.body_layout.addWidget(self.step_card)

        self.board_layout.addWidget(self.board_interface)
        self.board_layout.addLayout(self.button_layout)

        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(10, 5, 10, 10)
        self.body_layout.setSpacing(5)
        self.board_layout.setSpacing(5)

        self.setLayout(self.main_layout)

    def initTab(self):
        # add tab
        self.addTab(FluentIcon.PEOPLE, 'Game #0')
        self.board_interface.setCurrentWidget(self.findChild(BoardWidget, "Game #0"))

        self.tab_bar.setMovable(False)
        self.tab_bar.setTabMaximumWidth(200)
        self.tab_bar.setTabShadowEnabled(False)
        self.tab_bar.setTabSelectedBackgroundColor(QColor(255, 255, 255, 125), QColor(255, 255, 255, 50))
        self.tab_bar.itemLayout.setContentsMargins(0, 0, 0, 0)
        self.tab_bar.itemLayout.setSpacing(5)

        self.tab_bar.addButton.deleteLater()
        self.tab_bar.addButton = TransparentDropDownToolButton(FluentIcon.ADD, self)

        self.menu = RoundMenu(parent=self)
        self.menu.addAction(Action(FluentIcon.PEOPLE, 'Friend', triggered=self.onAddFriendTab, shortcut='Ctrl+N'))
        submenu = RoundMenu("Robot", self)
        submenu.setIcon(FluentIcon.ROBOT)
        submenu.addActions([
            Action(FluentIcon.ROBOT, 'Random', triggered=lambda: self.onAddRobotTab("Random")),
            Action(FluentIcon.ROBOT, 'Max Territory', triggered=lambda: self.onAddRobotTab("Max Territory")),
            Action(FluentIcon.ROBOT, 'Max Diff Sigmoid Territory',
                   triggered=lambda: self.onAddRobotTab("Max Diff Sigmoid Territory")),
            Action(FluentIcon.ROBOT, 'Max Percent Sigmoid Territory',
                   triggered=lambda: self.onAddRobotTab("Max Percent Sigmoid Territory")),
            Action(FluentIcon.ROBOT, 'MiniMax',
                   triggered=lambda: self.onAddRobotTab("MiniMax")),
        ])

        self.menu.addMenu(submenu)
        self.menu.addAction(Action(MyFluentIcon.CHIP, 'MCTS', triggered=self.onAddMCTSTab))
        self.menu.addAction(Action(MyFluentIcon.CHIP, 'Minimax', triggered=self.onAddMinimaxTab))

        self.tab_bar.addButton.setMenu(self.menu)
        self.tab_bar.addButton.setIconSize(QSize(12, 12))
        self.tab_bar.widgetLayout.addWidget(self.tab_bar.addButton, 0, Qt.AlignLeft)

        self.tab_bar.currentChanged.connect(self.onTabChanged)
        self.tab_bar.tabCloseRequested.connect(self.onTabClosed)
        self.tab_bar.addButton.clicked.connect(self.onAddButtonClicked)

    def connectButtons(self):
        try:
            self.load_button.clicked.disconnect()
            self.first_button.clicked.disconnect()
            self.previous_button.clicked.disconnect()
            self.next_button.clicked.disconnect()
            self.last_button.clicked.disconnect()
            self.save_button.clicked.disconnect()

            self.board_interface.currentWidget().onNotFirstMove.disconnect()
            self.board_interface.currentWidget().onNotLastMove.disconnect()
            self.board_interface.currentWidget().onSaveAvailable.disconnect()
            self.board_interface.currentWidget().onHistoryChanged.disconnect()

        except RuntimeError:
            pass

        self.load_button.clicked.connect(self.board_interface.currentWidget().onLoad)
        self.first_button.clicked.connect(self.board_interface.currentWidget().onFirst)
        self.previous_button.clicked.connect(self.board_interface.currentWidget().onPrevious)
        self.next_button.clicked.connect(self.board_interface.currentWidget().onNext)
        self.last_button.clicked.connect(self.board_interface.currentWidget().onLast)
        self.save_button.clicked.connect(self.board_interface.currentWidget().onSave)

        self.board_interface.currentWidget().onNotFirstMove.connect(self.first_button.setEnabled)
        self.board_interface.currentWidget().onNotFirstMove.connect(self.previous_button.setEnabled)
        self.board_interface.currentWidget().onNotLastMove.connect(self.next_button.setEnabled)
        self.board_interface.currentWidget().onNotLastMove.connect(self.last_button.setEnabled)
        self.board_interface.currentWidget().onSaveAvailable.connect(self.save_button.setEnabled)
        self.board_interface.currentWidget().onHistoryChanged.connect(self.refreshSteps)
        self.board_interface.currentWidget().onStepChanged.connect(self.refreshActiveStep)

    def initShortcuts(self):
        self.short_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.short_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.short_up = QShortcut(QKeySequence(Qt.Key_Up), self)
        self.short_down = QShortcut(QKeySequence(Qt.Key_Down), self)

        self.short_ctrl_n = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_N), self)
        self.short_ctrl_o = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_O), self)
        self.short_ctrl_s = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_S), self)

    def connectShortcuts(self):
        try:
            self.short_left.activated.disconnect()
            self.short_right.activated.disconnect()
            self.short_up.activated.disconnect()
            self.short_down.activated.disconnect()
            self.short_ctrl_n.activated.disconnect()
            self.short_ctrl_o.activated.disconnect()
            self.short_ctrl_s.activated.disconnect()
        except RuntimeError:
            pass

        self.short_left.activated.connect(self.board_interface.currentWidget().onPrevious)
        self.short_right.activated.connect(self.board_interface.currentWidget().onNext)
        self.short_up.activated.connect(self.board_interface.currentWidget().onFirst)
        self.short_down.activated.connect(self.board_interface.currentWidget().onLast)

        self.short_ctrl_n.activated.connect(self.onAddFriendTab)
        self.short_ctrl_o.activated.connect(self.board_interface.currentWidget().onLoad)
        self.short_ctrl_s.activated.connect(self.board_interface.currentWidget().onSave)

    def onStepClicked(self, index):
        self.board_interface.currentWidget().setCurrentStep(index + 1)

    def refreshSteps(self, history):
        self.step_card.addSteps(history)

    def refreshActiveStep(self, index):
        if index == 0:
            self.step_card.setAllInactive()
        else:
            self.step_card.setActiveStep(index - 1)

    def onAddFriendTab(self):
        self.addTab(FluentIcon.PEOPLE)

    def onAddRobotTab(self, robot):
        self.addTab(FluentIcon.ROBOT)
        self.board_interface.currentWidget().onSelectRobot(robot)

    def onAddMCTSTab(self):
        self.addTab(MyFluentIcon.CHIP)
        self.board_interface.currentWidget().onEnableMCTS()

    def onAddMinimaxTab(self):
        self.addTab(MyFluentIcon.CHIP)
        self.board_interface.currentWidget().onEnableMinimax()

    def addTab(self, icon, text=""):
        if not text:
            text = f'Game #{max([int(tab.routeKey().split("#")[1]) for tab in self.tab_bar.items]) + 1}'
        self.tab_bar.addTab(text, text, icon)
        self.board_interface.addWidget(BoardWidget(text, self))
        self.board_interface.setCurrentWidget(self.findChild(BoardWidget, text.replace(' ', '-')))
        self.tab_bar.setCurrentTab(text)
        self.connectButtons()
        self.connectShortcuts()
        self.refreshSteps(self.board_interface.currentWidget().history)

    def onTabChanged(self, index: int):
        objectName = self.tab_bar.currentTab().routeKey().replace(' ', '-')
        self.board_interface.setCurrentWidget(self.findChild(BoardWidget, objectName))
        self.connectButtons()
        self.connectShortcuts()
        self.refreshSteps(self.board_interface.currentWidget().history)

    def onTabClosed(self, index: int):
        if self.tab_bar.count() == 1:
            InfoBar.error(
                title="Error",
                content="You cannot close the last tab.",
                parent=self
            )
            return

        self.tab_bar.removeTab(index)
        self.board_interface.removeWidget(self.board_interface.widget(index))

    def onAddButtonClicked(self):
        self.menu.view.itemWidget(self.menu.view.item(1)).setFixedWidth(110)
