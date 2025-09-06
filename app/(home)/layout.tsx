import { HomeLayout } from "fumadocs-ui/layouts/home";
import { baseOptions } from "@/lib/layout.shared";
import {
  NavbarMenu,
  NavbarMenuContent,
  NavbarMenuLink,
  NavbarMenuTrigger,
} from "fumadocs-ui/layouts/home/navbar";
import Link from "fumadocs-core/link";
import Image from "next/image";
// import Preview from "@/public/banner.png";
import { Book, ComponentIcon, Pencil, PlusIcon, Server } from "lucide-react";

export default function Layout({ children }: LayoutProps<"/">) {
  return (
    <HomeLayout
      {...baseOptions()}
      links={[
        {
          type: "menu",
          on: "menu",
          text: "Documentation",
          items: [],
        },
        {
          type: "custom",
          on: "nav",
          children: (
            <NavbarMenu>
              <NavbarMenuTrigger>
                <Link href="/docs">Documentacion</Link>
              </NavbarMenuTrigger>
            </NavbarMenu>
          ),
        },
        // {
        //   type: "menu",
        //   on: "menu",
        //   text: "Acerca de mi",
        //   items: [],
        // },
        // {
        //   type: "custom",
        //   on: "nav",
        //   children: (
        //     <NavbarMenu>
        //       <NavbarMenuTrigger>
        //         <Link href="/acerca-de-mi">Acerca de mi</Link>
        //       </NavbarMenuTrigger>
        //     </NavbarMenu>
        //   ),
        // },
      ]}
    >
      {children}
    </HomeLayout>
  );
}
