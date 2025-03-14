from multivariate_view.app.app import App


def main():
    app = App()
    app.server.start(timeout=0)


if __name__ == '__main__':
    main()
